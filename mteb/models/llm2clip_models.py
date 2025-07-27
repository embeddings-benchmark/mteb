from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer, CLIPImageProcessor

from mteb.encoder_interface import PromptType
from mteb.model_meta import ModelMeta
from mteb.requires_package import requires_image_dependencies, requires_package

MODEL2PROCESSOR = {
    "microsoft/LLM2CLIP-Openai-L-14-336": "openai/clip-vit-large-patch14-336",
    "microsoft/LLM2CLIP-Openai-B-16": "openai/clip-vit-base-patch16",
    "microsoft/LLM2CLIP-Openai-L-14-224": "openai/clip-vit-large-patch14",
}


def llm2clip_loader(**kwargs):
    model_name = kwargs.get("model_name", "LLM2CLIP")
    requires_package(
        llm2clip_loader, "llm2vec", model_name, "pip install 'mteb[llm2vec]'"
    )
    from llm2vec import LLM2Vec

    class LLM2CLIPWrapper:
        def __init__(
            self,
            model_name: str = "microsoft/LLM2CLIP-Openai-L-14-336",
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
            llm_model.config._name_or_path = "meta-llama/Meta-Llama-3-8B-Instruct"  #  Workaround for LLM2VEC. constant.
            self.l2v = LLM2Vec(
                llm_model,
                tokenizer,
                pooling_mode="mean",
                max_length=512,
                doc_max_length=512,
            )

        def get_text_embeddings(
            self,
            texts: list[str],
            *,
            task_name: str | None = None,
            prompt_type: PromptType | None = None,
            batch_size: int = 32,
            **kwargs: Any,
        ):
            all_text_embeddings = []

            with torch.no_grad(), torch.amp.autocast("cuda"):
                for i in tqdm(range(0, len(texts), batch_size)):
                    batch_texts = texts[i : i + batch_size]
                    text_features = self.l2v.encode(
                        batch_texts, convert_to_tensor=True
                    ).to(self.device)
                    text_features = self.model.get_text_features(text_features)
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                    all_text_embeddings.append(text_features.cpu().to(torch.float32))

            all_text_embeddings = torch.cat(all_text_embeddings, dim=0)

            return all_text_embeddings

        def get_image_embeddings(
            self,
            images: list[Image.Image] | DataLoader,
            *,
            task_name: str | None = None,
            prompt_type: PromptType | None = None,
            batch_size: int = 32,
            **kwargs: Any,
        ):
            import torchvision.transforms.functional as F

            all_image_embeddings = []
            if isinstance(images, DataLoader):
                with torch.no_grad(), torch.amp.autocast("cuda"):
                    for batch in tqdm(images):
                        input_pixels = self.processor(
                            images=[F.to_pil_image(b) for b in batch],
                            return_tensors="pt",
                        ).pixel_values.to(self.device)
                        image_features = self.model.get_image_features(input_pixels)
                        image_features /= image_features.norm(dim=-1, keepdim=True)
                        all_image_embeddings.append(
                            image_features.cpu().to(torch.float32)
                        )
            else:
                with torch.no_grad(), torch.cuda.amp.autocast():
                    for i in tqdm(range(0, len(images), batch_size)):
                        batch_images = images[i : i + batch_size]
                        input_pixels = self.processor(
                            images=batch_images, return_tensors="pt"
                        ).pixel_values.to(self.device)
                        image_features = self.model.get_image_features(input_pixels)
                        image_features /= image_features.norm(dim=-1, keepdim=True)
                        all_image_embeddings.append(
                            image_features.cpu().to(torch.float32)
                        )

            all_image_embeddings = torch.cat(all_image_embeddings, dim=0)
            return all_image_embeddings

        def calculate_probs(self, text_embeddings, image_embeddings):
            text_embeddings = text_embeddings / text_embeddings.norm(
                dim=-1, keepdim=True
            )
            image_embeddings = image_embeddings / image_embeddings.norm(
                dim=-1, keepdim=True
            )
            logits = torch.matmul(image_embeddings, text_embeddings.T)
            probs = (logits * 100).softmax(dim=-1)
            return probs

        def get_fused_embeddings(
            self,
            texts: list[str] = None,
            images: list[Image.Image] | DataLoader = None,
            fusion_mode="sum",
            **kwargs: Any,
        ):
            if texts is None and images is None:
                raise ValueError("Either texts or images must be provided")

            text_embeddings = None
            image_embeddings = None

            if texts is not None:
                text_embeddings = self.get_text_embeddings(texts, **kwargs)

            if images is not None:
                image_embeddings = self.get_image_embeddings(images, **kwargs)

            if text_embeddings is not None and image_embeddings is not None:
                if len(text_embeddings) != len(image_embeddings):
                    raise ValueError(
                        "The number of texts and images must have the same length"
                    )
                if fusion_mode == "sum":
                    fused_embeddings = text_embeddings + image_embeddings
                else:
                    # to do: add other fusion mode
                    raise ValueError(
                        f"fusion mode {fusion_mode} hasn't been implemented"
                    )
                return fused_embeddings
            elif text_embeddings is not None:
                return text_embeddings
            elif image_embeddings is not None:
                return image_embeddings

    return LLM2CLIPWrapper(**kwargs)


llm2clip_training_sets = {
    # CC3M
    # CC12M
    # YFCC15M
    # Recap-DataComp-1B(30M subset)
}

llm2clip_openai_l_14_336 = ModelMeta(
    loader=partial(
        llm2clip_loader,
        model_name="microsoft/LLM2CLIP-Openai-L-14-336",
    ),
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
    similarity_fn_name=None,
    use_instructions=True,
    training_datasets=llm2clip_training_sets,
)

## NOTE: https://huggingface.co/microsoft/LLM2CLIP-Openai-L-14-224/discussions/1
llm2clip_openai_l_14_224 = ModelMeta(
    loader=partial(
        llm2clip_loader,
        model_name="microsoft/LLM2CLIP-Openai-L-14-224",
    ),
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
    similarity_fn_name=None,
    use_instructions=True,
    training_datasets=llm2clip_training_sets,
)

llm2clip_openai_b_16 = ModelMeta(
    loader=partial(
        llm2clip_loader,
        model_name="microsoft/LLM2CLIP-Openai-B-16",
    ),
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
    similarity_fn_name=None,
    use_instructions=True,
    training_datasets=llm2clip_training_sets,
)


if __name__ == "__main__":
    m = llm2clip_loader()
    emb = m.get_text_embeddings(
        texts=["what is going on blah?", "this is a test for this model."]
    )
