from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Unpack

import torch
from tqdm.auto import tqdm

from mteb._requires_package import (
    suggest_package,
)
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.types import PromptType

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput, EncodeKwargs

logger = logging.getLogger(__name__)

VLM2VEC_CITATION = """@article{jiang2024vlm2vec,
  title={VLM2Vec: Training Vision-Language Models for Massive Multimodal Embedding Tasks},
  author={Jiang, Ziyan and Meng, Rui and Yang, Xinyi and Yavuz, Semih and Zhou, Yingbo and Chen, Wenhu},
  journal={arXiv preprint arXiv:2410.05160},
  year={2024}
}"""


class VLM2VecWrapper(AbsEncoder):
    """Adapted from https://github.com/TIGER-AI-Lab/VLM2Vec/blob/main/src/model.py"""

    def __init__(
        self,
        model_name: str = "TIGER-Lab/VLM2Vec-LoRA",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs,
    ):
        if suggest_package(
            self,
            "flash_attn",
            model_name,
            "pip install flash-attn --no-build-isolation",
        ):
            pass

        from peft import LoraConfig, PeftModel
        from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor

        self.pooling = "last"
        self.normalize = True
        self.temperature = 1.0
        self.hidden_size = 4096
        self.device = device

        # Loading the base model
        base_model_name = "microsoft/Phi-3.5-vision-instruct"
        config = AutoConfig.from_pretrained(base_model_name, trust_remote_code=True)
        config.use_cache = False
        config.padding_side = "right"

        checkpoint_path = model_name if model_name else base_model_name
        base_model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            config=config,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        base_model.padding_side = "right"

        # Building the model on top of the base
        if "LoRA" in model_name:
            lora_config = LoraConfig.from_pretrained(checkpoint_path)
            lora_model = PeftModel.from_pretrained(
                base_model, checkpoint_path, config=lora_config
            )
            merged_model = lora_model.merge_and_unload()
            model = merged_model.to(torch.bfloat16)  # propagate dtype.
        else:
            model = base_model.to(torch.bfloat16)

        model.eval()
        model.to(device)
        self.mdl = model

        self.processor = AutoProcessor.from_pretrained(
            base_model_name,
            trust_remote_code=True,
            num_crops=4,
        )

    def encode_input(self, input):
        hidden_states = self.mdl(**input, return_dict=True, output_hidden_states=True)
        hidden_states = hidden_states.hidden_states[-1]
        pooled_output = self._pooling(hidden_states, input["attention_mask"])
        return pooled_output

    def _pooling(self, last_hidden_state, attention_mask):
        if self.pooling == "last":
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_state.shape[0]
            reps = last_hidden_state[
                torch.arange(batch_size, device=last_hidden_state.device),
                sequence_lengths,
            ]
        else:
            raise NotImplementedError
        if self.normalize:
            reps = torch.nn.functional.normalize(reps, p=2, dim=-1)
        return reps

    # reference: https://github.com/TIGER-AI-Lab/VLM2Vec/blob/main/src/collator.py
    def get_image_embeddings(
        self,
        images: DataLoader[BatchedInput],
        show_progress_bar: bool = True,
        **kwargs: Any,
    ):
        text = "<|image_1|> Represent the given image."
        all_image_embeddings = []

        with torch.no_grad():
            for batch in tqdm(
                images, disable=not show_progress_bar, desc="Image Encoding"
            ):
                input_ids, pixel_values, image_sizes = [], [], []
                for b in batch["image"]:
                    inputs = self.processor(
                        text,
                        b,
                        return_tensors="pt",
                        max_length=256,
                        truncation=True,
                    )
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    input_ids.append(inputs["input_ids"].squeeze(0).unsqueeze(1))
                    pixel_values.append(inputs["pixel_values"])
                    image_sizes.append(inputs["image_sizes"])

                input_ids = torch._C._nn.pad_sequence(
                    input_ids,
                    batch_first=True,
                    padding_value=self.processor.tokenizer.pad_token_id,
                ).squeeze(2)
                attention_mask = input_ids.ne(self.processor.tokenizer.pad_token_id)

                pixel_values = torch.cat(pixel_values, dim=0)
                image_sizes = torch.cat(image_sizes, dim=0)
                inputs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "pixel_values": pixel_values,
                    "image_sizes": image_sizes,
                }

                image_outputs = self.encode_input(inputs)
                all_image_embeddings.append(image_outputs.cpu().to(torch.float32))

        all_image_embeddings = torch.cat(all_image_embeddings, dim=0)
        return all_image_embeddings

    def get_text_embeddings(
        self,
        texts: DataLoader[BatchedInput],
        show_progress_bar: bool = True,
        **kwargs: Any,
    ):
        all_text_embeddings = []

        with torch.no_grad():
            for batch in tqdm(
                texts, disable=not show_progress_bar, desc="Text Encoding"
            ):
                input_ids = []
                for text in batch["text"]:
                    inputs = self.processor(
                        text,
                        None,
                        return_tensors="pt",
                        max_length=256,
                        truncation=True,
                    )
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    input_ids.append(inputs["input_ids"].squeeze(0).unsqueeze(1))

                input_ids = torch._C._nn.pad_sequence(
                    input_ids,
                    batch_first=True,
                    padding_value=self.processor.tokenizer.pad_token_id,
                ).squeeze(2)
                attention_mask = input_ids.ne(self.processor.tokenizer.pad_token_id)
                inputs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                }

                text_outputs = self.encode_input(inputs)
                all_text_embeddings.append(text_outputs.cpu().to(torch.float32))

        all_text_embeddings = torch.cat(all_text_embeddings, dim=0)
        return all_text_embeddings

    def encode(
        self,
        inputs: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        **kwargs: Unpack[EncodeKwargs],
    ) -> Array:
        if "text" in inputs.dataset.features and "image" in inputs.dataset.features:
            all_fused_embeddings = []

            with torch.no_grad():
                for batch in inputs:
                    input_ids, pixel_values, image_sizes = [], [], []
                    batch_text = batch["text"]
                    batch_image = batch["image"]
                    for item_image, item_text in zip(batch_image, batch_text):
                        inputs = self.processor(
                            f"<|image_1|> Represent the given image with the following question: {item_text}",
                            item_image,
                            return_tensors="pt",
                            max_length=256,
                            truncation=True,
                        )
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                        input_ids.append(inputs["input_ids"].squeeze(0).unsqueeze(1))
                        pixel_values.append(inputs["pixel_values"])
                        image_sizes.append(inputs["image_sizes"])

                    input_ids = torch._C._nn.pad_sequence(
                        input_ids,
                        batch_first=True,
                        padding_value=self.processor.tokenizer.pad_token_id,
                    ).squeeze(2)
                    attention_mask = input_ids.ne(self.processor.tokenizer.pad_token_id)

                    pixel_values = torch.cat(pixel_values, dim=0)
                    image_sizes = torch.cat(image_sizes, dim=0)
                    inputs = {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "pixel_values": pixel_values,
                        "image_sizes": image_sizes,
                    }

                    outputs = self.encode_input(inputs)
                    all_fused_embeddings.append(outputs.cpu().to(torch.float32))

                fused_embeddings = torch.cat(all_fused_embeddings, dim=0)
                return fused_embeddings
        elif "image" in inputs.dataset.features:
            image_embeddings = self.get_image_embeddings(inputs, **kwargs)
            return image_embeddings
        elif "text" in inputs.dataset.features:
            text_embeddings = self.get_text_embeddings(inputs, **kwargs)
            return text_embeddings
        raise ValueError


_VLM2VEC2_VIDEO_TOKEN = "<|video_pad|>"  # noqa: S105
_VLM2VEC2_IMAGE_TOKEN = "<|image_pad|>"  # noqa: S105

_VLM2VEC2_TASK_PROMPTS: dict[str, dict[str, str]] = {
    # --- Video retrieval ---
    "DiDeMoT2VRetrieval": {
        "query": "Find a video that includes the following described scenes:",
        "document": "Understand the content of the provided video.",
    },
    "DiDeMoV2TRetrieval": {
        "query": "Understand the content of the provided video.",
        "document": "Find a video that includes the following described scenes:",
    },
    "MSRVTTT2V": {
        "query": "Find a video that contains the following visual content:",
        "document": "Understand the content of the provided video.",
    },
    "MSRVTTV2T": {
        "query": "Understand the content of the provided video.",
        "document": "Find a video that contains the following visual content:",
    },
    "MSVDT2VRetrieval": {
        "query": "Find the video snippet that corresponds to the given summary:",
        "document": "Understand the content of the provided video.",
    },
    "MSVDV2TRetrieval": {
        "query": "Understand the content of the provided video.",
        "document": "Find the video snippet that corresponds to the given summary:",
    },
    "VATEXT2VRetrieval": {
        "query": "Select a video that fits the description provided:",
        "document": "Understand the content of the provided video.",
    },
    "VATEXV2TRetrieval": {
        "query": "Understand the content of the provided video.",
        "document": "Select a video that fits the description provided:",
    },
    "YouCook2T2VRetrieval": {
        "query": "Find a video that demonstrates the following action while making a recipe:",
        "document": "Understand the content of the provided video.",
    },
    "YouCook2V2TRetrieval": {
        "query": "Understand the content of the provided video.",
        "document": "Find a video that demonstrates the following action while making a recipe:",
    },
    # --- Image retrieval (VisRAG) ---
    "VisRAGRetChartQA": {
        "query": "Find a document image that matches the given query:",
        "document": "Understand the content of the provided document image.",
    },
    "VisRAGRetInfoVQA": {
        "query": "Find a document image that matches the given query:",
        "document": "Understand the content of the provided document image.",
    },
    "VisRAGRetMPDocVQA": {
        "query": "Find a document image that matches the given query:",
        "document": "Understand the content of the provided document image.",
    },
    "VisRAGRetArxivQA": {
        "query": "Find a document image that matches the given query:",
        "document": "Understand the content of the provided document image.",
    },
    "VisRAGRetPlotQA": {
        "query": "Find a document image that matches the given query:",
        "document": "Understand the content of the provided document image.",
    },
    "VisRAGRetSlideVQA": {
        "query": "Find a document image that matches the given query:",
        "document": "Understand the content of the provided document image.",
    },
    # --- Image retrieval (Vidore) ---
    **{
        name: {
            "query": "Find a document image that matches the given query:",
            "document": "Understand the content of the provided document image.",
        }
        for name in (
            "VidoreArxivQARetrieval",
            "VidoreDocVQARetrieval",
            "VidoreInfoVQARetrieval",
            "VidoreTabfquadRetrieval",
            "VidoreTatdqaRetrieval",
            "VidoreShiftProjectRetrieval",
            "VidoreSyntheticDocQAAIRetrieval",
            "VidoreSyntheticDocQAEnergyRetrieval",
            "VidoreSyntheticDocQAGovernmentReportsRetrieval",
            "VidoreSyntheticDocQAHealthcareIndustryRetrieval",
            "Vidore2ESGReportsRetrieval",
            "Vidore2EconomicsReportsRetrieval",
            "Vidore2BioMedicalLecturesRetrieval",
            "Vidore2ESGReportsHLRetrieval",
            "Vidore3FinanceEnRetrieval",
            "Vidore3FinanceEnRetrieval.v2",
            "Vidore3FinanceFrRetrieval",
            "Vidore3FinanceFrRetrieval.v2",
            "Vidore3IndustrialRetrieval",
            "Vidore3IndustrialRetrieval.v2",
            "Vidore3PharmaceuticalsRetrieval",
            "Vidore3PharmaceuticalsRetrieval.v2",
            "Vidore3ComputerScienceRetrieval",
            "Vidore3ComputerScienceRetrieval.v2",
            "Vidore3HrRetrieval",
            "Vidore3HrRetrieval.v2",
            "Vidore3EnergyRetrieval",
            "Vidore3EnergyRetrieval.v2",
            "Vidore3PhysicsRetrieval",
            "Vidore3PhysicsRetrieval.v2",
            "Vidore3NuclearRetrieval",
            "Vidore3NuclearRetrieval.v2",
            "Vidore3TelecomRetrieval",
            "Vidore3TelecomRetrieval.v2",
            "KoVidore2CybersecurityRetrieval",
            "KoVidore2EconomicRetrieval",
            "KoVidore2EnergyRetrieval",
            "KoVidore2HrRetrieval",
        )
    },
}


class VLM2VEC2Wrapper(AbsEncoder):
    def __init__(
        self,
        model: str,
        revision: str | None = None,
        *,
        device: str | None = None,
        embed_dim: int | None = None,
        **kwargs,
    ) -> None:
        self.device = device
        from peft import PeftModel
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        base_model_name = "Qwen/Qwen2-VL-2B-Instruct"
        self.processor = AutoProcessor.from_pretrained(base_model_name)
        self.processor.padding_side = "left"

        base_model = Qwen2VLForConditionalGeneration.from_pretrained(
            base_model_name, **kwargs
        )
        self.model = PeftModel.from_pretrained(base_model, model, revision=revision)
        self.model.merge_and_unload()
        self.model.to(device)
        self.model.eval()

    @staticmethod
    def _pooling(
        last_hidden_state: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
        batch_size = last_hidden_state.shape[0]
        if left_padding:
            reps = last_hidden_state[torch.arange(batch_size), -1, :]
        else:
            eos_indices = attention_mask.sum(dim=1) - 1
            reps = last_hidden_state[
                torch.arange(batch_size, device=last_hidden_state.device), eos_indices
            ]
        return torch.nn.functional.normalize(reps, p=2, dim=-1)

    @torch.inference_mode()
    def encode(  # noqa: PLR0914
        self,
        inputs: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        **kwargs: Unpack[EncodeKwargs],
    ) -> Array:
        features = inputs.dataset.features
        has_text = "text" in features
        has_image = "image" in features
        has_video = "video" in features

        show_progress_bar = kwargs.get("show_progress_bar", True)

        all_embeddings: list[torch.Tensor] = []

        for batch in tqdm(inputs, disable=not show_progress_bar, desc="Encoding"):
            batch_embeddings: list[torch.Tensor] = []
            batch_size = len(next(iter(batch.values())))

            for i in range(batch_size):
                tokens = []
                images = [batch["image"][i]] if has_image else None
                videos = [batch["video"][i]] if has_video else None

                if has_image:
                    tokens.append(_VLM2VEC2_IMAGE_TOKEN)
                if has_video:
                    tokens.append(_VLM2VEC2_VIDEO_TOKEN)

                text = batch["text"][i] if has_text else None
                task_prompts = _VLM2VEC2_TASK_PROMPTS.get(task_metadata.name, {})
                prompt_key = "query" if prompt_type == PromptType.query else "document"
                instruction = task_prompts.get(prompt_key, "")
                prefix = " ".join(tokens)
                prompt = (
                    f"{prefix} {instruction} {text}".strip()
                    if text
                    else f"{prefix} {instruction}".strip()
                )

                proc_inputs = self.processor(
                    text=prompt,
                    images=images,
                    videos=videos,
                    return_tensors="pt",
                )
                proc_inputs = {k: v.to(self.device) for k, v in proc_inputs.items()}

                if has_video:
                    proc_inputs["pixel_values_videos"] = proc_inputs[
                        "pixel_values_videos"
                    ].unsqueeze(0)
                    proc_inputs["video_grid_thw"] = proc_inputs[
                        "video_grid_thw"
                    ].unsqueeze(0)

                output = self.model(
                    **proc_inputs, return_dict=True, output_hidden_states=True
                )
                hidden_states = output.hidden_states[-1]
                emb = self._pooling(hidden_states, proc_inputs["attention_mask"])
                batch_embeddings.append(emb.cpu().to(torch.float32))

            if batch_embeddings:
                all_embeddings.append(torch.cat(batch_embeddings, dim=0))

        return torch.cat(all_embeddings, dim=0)


vlm2vec_training_datasets = set(
    # MMEB-train
)

vlm2vec2_training_datasets = set(
    # MMEBv2-train
)

vlm2vec_lora = ModelMeta(
    loader=VLM2VecWrapper,
    name="TIGER-Lab/VLM2Vec-LoRA",
    model_type=["dense"],
    languages=["eng-Latn"],
    revision="7403b6327958071c1e33c822c7453adadccc7298",
    release_date="2024-10-08",
    modalities=["image", "text"],
    n_parameters=4161891328,
    n_embedding_parameters=98500608,
    memory_usage_mb=None,
    max_tokens=131072,
    embed_dim=3072,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/TIGER-AI-Lab/VLM2Vec",
    public_training_data="https://huggingface.co/datasets/TIGER-Lab/MMEB-train",
    framework=["PyTorch", "Transformers"],
    reference="https://huggingface.co/TIGER-Lab/VLM2Vec-LoRA",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=True,
    training_datasets=vlm2vec_training_datasets,
    citation=VLM2VEC_CITATION,
    extra_requirements_groups=["peft"],
)

vlm2vec_full = ModelMeta(
    loader=VLM2VecWrapper,
    name="TIGER-Lab/VLM2Vec-Full",
    model_type=["dense"],
    languages=["eng-Latn"],
    revision="e9afa98002097ac2471827ba23ea1f2ddd229480",
    release_date="2024-10-08",
    modalities=["image", "text"],
    n_parameters=4146621440,
    n_embedding_parameters=98500608,
    memory_usage_mb=7909,
    max_tokens=131072,
    embed_dim=3072,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/TIGER-AI-Lab/VLM2Vec",
    public_training_data="https://huggingface.co/TIGER-Lab/VLM2Vec-Full",
    framework=["PyTorch", "Transformers", "safetensors"],
    reference="https://huggingface.co/TIGER-Lab/VLM2Vec-Full",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=True,
    training_datasets=vlm2vec_training_datasets,
    citation=VLM2VEC_CITATION,
    extra_requirements_groups=["peft"],
)

vlm2vec2 = ModelMeta(
    loader=VLM2VEC2Wrapper,
    name="VLM2Vec/VLM2Vec-V2.0",
    revision="e39ff079b8275ef876d3656da8c0bddbff3c4dde",
    release_date="2025-04-30",
    languages=["eng-Latn"],
    n_parameters=2208985600,
    n_embedding_parameters=233_373_696,
    memory_usage_mb=4213,
    max_tokens=32768,
    embed_dim=1536,
    license="apache-2.0",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["Transformers", "safetensors"],
    reference="https://huggingface.co/VLM2Vec/VLM2Vec-V2.0",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=None,
    training_datasets=vlm2vec_training_datasets | vlm2vec2_training_datasets,
    adapted_from="Qwen/Qwen2-VL-2B-Instruct",
    superseded_by=None,
    modalities=["text", "image", "video"],
    model_type=["dense"],
    citation="""
@misc{meng2025vlm2vecv2advancingmultimodalembedding,
    title={VLM2Vec-V2: Advancing Multimodal Embedding for Videos, Images, and Visual Documents},
    author={Rui Meng and Ziyan Jiang and Ye Liu and Mingyi Su and Xinyi Yang and Yuepeng Fu and Can Qin and Zeyuan Chen and Ran Xu and Caiming Xiong and Yingbo Zhou and Wenhu Chen and Semih Yavuz},
    year={2025},
    eprint={2507.04590},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2507.04590},
}""",
    contacts=None,
    output_dtypes=None,
    extra_requirements_groups=["peft"],
)
