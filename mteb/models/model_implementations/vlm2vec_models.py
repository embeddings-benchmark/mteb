import logging
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from mteb._requires_package import (
    requires_image_dependencies,
    requires_package,
    suggest_package,
)
from mteb.abstasks.task_metadata import TaskMetadata
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.types import Array, BatchedInput, PromptType

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
        requires_image_dependencies()
        if suggest_package(
            self,
            "flash_attn",
            model_name,
            "pip install flash-attn --no-build-isolation",
        ):
            import flash_attn  # noqa

        requires_package(self, "peft", model_name, "pip install 'mteb[peft]'")
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
        **kwargs: Any,
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


vlm2vec_training_datasets = set(
    # MMEB-train
)

vlm2vec_lora = ModelMeta(
    loader=VLM2VecWrapper,
    name="TIGER-Lab/VLM2Vec-LoRA",
    languages=["eng-Latn"],
    revision="7403b6327958071c1e33c822c7453adadccc7298",
    release_date="2024-10-08",
    modalities=["image", "text"],
    n_parameters=None,
    memory_usage_mb=None,
    max_tokens=131072,
    embed_dim=3072,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/TIGER-AI-Lab/VLM2Vec",
    public_training_data="https://huggingface.co/datasets/TIGER-Lab/MMEB-train",
    framework=["PyTorch"],
    reference="https://huggingface.co/TIGER-Lab/VLM2Vec-LoRA",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=True,
    training_datasets=vlm2vec_training_datasets,
    citation=VLM2VEC_CITATION,
)

vlm2vec_full = ModelMeta(
    loader=VLM2VecWrapper,
    name="TIGER-Lab/VLM2Vec-Full",
    languages=["eng-Latn"],
    revision="e9afa98002097ac2471827ba23ea1f2ddd229480",
    release_date="2024-10-08",
    modalities=["image", "text"],
    n_parameters=4_150_000_000,
    memory_usage_mb=7909,
    max_tokens=131072,
    embed_dim=3072,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/TIGER-AI-Lab/VLM2Vec",
    public_training_data="https://huggingface.co/TIGER-Lab/VLM2Vec-Full",
    framework=["PyTorch"],
    reference="https://huggingface.co/TIGER-Lab/VLM2Vec-Full",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=True,
    training_datasets=vlm2vec_training_datasets,
    citation=VLM2VEC_CITATION,
)
