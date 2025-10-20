from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from mteb._requires_package import requires_image_dependencies, requires_package
from mteb.abstasks.task_metadata import TaskMetadata
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.types import Array, BatchedInput, PromptType

LLM2CLIP_CITATION = """@misc{huang2024llm2clippowerfullanguagemodel,
  title={LLM2CLIP: Powerful Language Model Unlock Richer Visual Representation},
  author={Weiquan Huang and Aoqi Wu and Yifan Yang and Xufang Luo and Yuqing Yang and Liang Hu and Qi Dai and Xiyang Dai and Dongdong Chen and Chong Luo and Lili Qiu},
  year={2024},
  eprint={2411.04997},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2411.04997}
}"""

MODEL2PROCESSOR = {
    "microsoft/LLM2CLIP-Openai-L-14-336": "openai/clip-vit-large-patch14-336",
    "microsoft/LLM2CLIP-Openai-B-16": "openai/clip-vit-base-patch16",
    "microsoft/LLM2CLIP-Openai-L-14-224": "openai/clip-vit-large-patch14",
}


def llm2clip_loader(model_name, **kwargs):
    from transformers import AutoConfig, AutoModel, AutoTokenizer, CLIPImageProcessor

    requires_package(
        llm2clip_loader, "llm2vec", model_name, "pip install 'mteb[llm2vec]'"
    )
    from llm2vec import LLM2Vec

    class LLM2CLIPAbsEncoder(AbsEncoder):
        def __init__(
            self,
            model_name: str,
            device: str = "cuda" if torch.cuda.is_available() else "cpu",
            **kwargs: Any,
        ):
            requires_image_dependencies()

            if model_name not in MODEL2PROCESSOR:
                raise Exception(
                    f"This model {model_name} is not in the supported mode list: {list(MODEL2PROCESSOR.keys())}."
                )

            self.device = device
            from huggingface_hub import snapshot_download

            model_folder_path = snapshot_download(
                repo_id=model_name, allow_patterns=["*.json", "*.safetensors", "*.py"]
            )
            snapshot_download(
                repo_id=MODEL2PROCESSOR[model_name],
                allow_patterns=["*.json", "*.safetensors", "*.py"],
            )
            model_name_or_path = Path(model_folder_path)
            self.processor = CLIPImageProcessor.from_pretrained(
                MODEL2PROCESSOR[model_name]
            )
            self.model = (
                AutoModel.from_pretrained(
                    model_name_or_path,
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                )
                .to(self.device)
                .eval()
            )

            llm_model_name = (
                "microsoft/LLM2CLIP-Llama-3-8B-Instruct-CC-Finetuned"  # constant
            )
            config = AutoConfig.from_pretrained(llm_model_name, trust_remote_code=True)
            llm_model = AutoModel.from_pretrained(
                llm_model_name,
                torch_dtype=torch.bfloat16,
                config=config,
                trust_remote_code=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
            llm_model.config._name_or_path = "meta-llama/Meta-Llama-3-8B-Instruct"  # Workaround for LLM2VEC. constant.
            self.l2v = LLM2Vec(
                llm_model,
                tokenizer,
                pooling_mode="mean",
                max_length=512,
                doc_max_length=512,
            )

        def get_text_embeddings(
            self,
            texts: DataLoader[BatchedInput],
            show_progress_bar: bool = True,
            **kwargs: Any,
        ):
            all_text_embeddings = []

            with torch.no_grad(), torch.amp.autocast("cuda"):
                for batch in tqdm(
                    texts, disable=not show_progress_bar, desc="Text Encoding"
                ):
                    text_features = self.l2v.encode(
                        batch["text"], convert_to_tensor=True
                    ).to(self.device)
                    text_features = self.model.get_text_features(text_features)
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                    all_text_embeddings.append(text_features.cpu().to(torch.float32))

            all_text_embeddings = torch.cat(all_text_embeddings, dim=0)

            return all_text_embeddings

        def get_image_embeddings(
            self,
            images: DataLoader[BatchedInput],
            show_progress_bar: bool = True,
            **kwargs: Any,
        ):
            all_image_embeddings = []

            with torch.no_grad(), torch.amp.autocast("cuda"):
                for batch in tqdm(
                    images, disable=not show_progress_bar, desc="Image Encoding"
                ):
                    input_pixels = self.processor(
                        images=batch["image"],
                        return_tensors="pt",
                    ).pixel_values.to(self.device)
                    image_features = self.model.get_image_features(input_pixels)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    all_image_embeddings.append(image_features.cpu().to(torch.float32))

            all_image_embeddings = torch.cat(all_image_embeddings, dim=0)
            return all_image_embeddings

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
            text_embeddings = None
            image_embeddings = None
            if "text" in inputs.dataset.features:
                text_embeddings = self.get_text_embeddings(inputs, **kwargs)
            if "image" in inputs.dataset.features:
                image_embeddings = self.get_image_embeddings(inputs, **kwargs)

            if text_embeddings is not None and image_embeddings is not None:
                if len(text_embeddings) != len(image_embeddings):
                    raise ValueError(
                        "The number of texts and images must have the same length"
                    )
                fused_embeddings = text_embeddings + image_embeddings
                return fused_embeddings
            elif text_embeddings is not None:
                return text_embeddings
            elif image_embeddings is not None:
                return image_embeddings
            raise ValueError

    return LLM2CLIPAbsEncoder(model_name, **kwargs)


llm2clip_training_sets = set(
    # CC3M
    # CC12M
    # YFCC15M
    # Recap-DataComp-1B(30M subset)
)

llm2clip_openai_l_14_336 = ModelMeta(
    loader=llm2clip_loader,  # type: ignore
    name="microsoft/LLM2CLIP-Openai-L-14-336",
    languages=["eng-Latn"],
    revision="92512331f393a003c3d98404677f991c188162c9",
    release_date="2024-11-07",
    modalities=["image", "text"],
    n_parameters=579_000_000,
    memory_usage_mb=None,
    max_tokens=None,
    embed_dim=1280,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/microsoft/LLM2CLIP",
    public_training_data=None,
    framework=["PyTorch"],
    reference="https://huggingface.co/microsoft/LLM2CLIP-Openai-L-14-336",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=True,
    training_datasets=llm2clip_training_sets,
    citation=LLM2CLIP_CITATION,
)

# NOTE: https://huggingface.co/microsoft/LLM2CLIP-Openai-L-14-224/discussions/1
llm2clip_openai_l_14_224 = ModelMeta(
    loader=llm2clip_loader,  # type: ignore
    name="microsoft/LLM2CLIP-Openai-L-14-224",
    languages=["eng-Latn"],
    revision="6b8a11a94ff380fa220dfefe73ac9293d2677575",
    release_date="2024-11-07",
    modalities=["image", "text"],
    n_parameters=578_000_000,
    memory_usage_mb=None,
    max_tokens=None,
    embed_dim=1280,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/microsoft/LLM2CLIP",
    public_training_data=None,
    framework=["PyTorch"],
    reference="https://huggingface.co/microsoft/LLM2CLIP-Openai-L-14-224",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=True,
    training_datasets=llm2clip_training_sets,
    citation=LLM2CLIP_CITATION,
)

llm2clip_openai_b_16 = ModelMeta(
    loader=llm2clip_loader,  # type: ignore
    name="microsoft/LLM2CLIP-Openai-B-16",
    languages=["eng-Latn"],
    revision="ecfb347eb3dcfeb2fbc2a2eae7de6ac5a001aaf8",
    release_date="2024-11-07",
    modalities=["image", "text"],
    n_parameters=361_000_000,
    memory_usage_mb=None,
    max_tokens=None,
    embed_dim=1280,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/microsoft/LLM2CLIP",
    public_training_data=None,
    framework=["PyTorch"],
    reference="https://huggingface.co/microsoft/LLM2CLIP-Openai-B-16",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=True,
    training_datasets=llm2clip_training_sets,
    citation=LLM2CLIP_CITATION,
)
