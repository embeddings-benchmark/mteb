from __future__ import annotations

from functools import partial
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

from mteb.encoder_interface import PromptType
from mteb.model_meta import ModelMeta


class SiglipModelWrapper:
    def __init__(
        self,
        model_name: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs: Any,
    ):
        self.model_name = model_name
        self.device = device
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_name)

    def preprocess(
        self,
        texts: list[str],
        images: list[Image.Image],
    ):
        return self.processor(
            text=texts, images=images, return_tensors="pt", padding=True
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

        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size)):
                batch_texts = texts[i : i + batch_size]
                inputs = self.processor(
                    text=batch_texts,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                text_outputs = self.model.get_text_features(**inputs)
                all_text_embeddings.append(text_outputs.cpu())

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
        all_image_embeddings = []

        if isinstance(images, DataLoader):
            with torch.no_grad():
                for batch in tqdm(images):
                    inputs = self.processor(
                        images=batch, return_tensors="pt", padding=True
                    )
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    image_outputs = self.model.get_image_features(**inputs)
                    all_image_embeddings.append(image_outputs.cpu())
        else:
            with torch.no_grad():
                for i in tqdm(range(0, len(images), batch_size)):
                    batch_images = images[i : i + batch_size]
                    batch_images = [
                        img.convert("RGB")
                        if isinstance(img, Image.Image) and img.mode != "RGB"
                        else img
                        for img in batch_images
                    ]
                    inputs = self.processor(
                        images=batch_images, return_tensors="pt", padding=True
                    )
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    image_outputs = self.model.get_image_features(**inputs)
                    all_image_embeddings.append(image_outputs.cpu())

        all_image_embeddings = torch.cat(all_image_embeddings, dim=0)
        return all_image_embeddings

    def calculate_probs(self, text_embeddings, image_embeddings):
        # normalized features
        image_embeddings = image_embeddings / image_embeddings.norm(
            p=2, dim=-1, keepdim=True
        )
        text_embeddings = text_embeddings / text_embeddings.norm(
            p=2, dim=-1, keepdim=True
        )

        # cosine similarity as logits
        logits_per_text = torch.matmul(
            text_embeddings, image_embeddings.t().to(text_embeddings.device)
        ) * self.model.logit_scale.exp().to(
            text_embeddings.device
        ) + self.model.logit_bias.to(text_embeddings.device)
        logits_per_image = logits_per_text.t()
        return logits_per_image

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
                raise ValueError(f"fusion mode {fusion_mode} hasn't been implemented")
            return fused_embeddings
        elif text_embeddings is not None:
            return text_embeddings
        elif image_embeddings is not None:
            return image_embeddings


siglip_training_datasets = {
    # WebLI https://arxiv.org/abs/2209.06794
}

siglip_so400m_patch14_224 = ModelMeta(
    loader=partial(
        SiglipModelWrapper,
        model_name="google/siglip-so400m-patch14-224",
    ),
    name="google/siglip-so400m-patch14-224",
    languages=["eng-Latn"],
    revision="d04cf29fca7b6374f74d8bea1969314492266b5e",
    release_date="2024-01-08",
    modalities=["image", "text"],
    n_parameters=877_000_000,
    memory_usage_mb=3347,
    max_tokens=16,
    embed_dim=1152,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/google-research/big_vision/blob/main/big_vision/trainers/proj/image_text/siglip.py",
    public_training_data=None,
    framework=["PyTorch"],
    reference="https://huggingface.co/google/siglip-so400m-patch14-224",
    similarity_fn_name=None,
    use_instructions=False,
    training_datasets=siglip_training_datasets,
)

siglip_so400m_patch14_384 = ModelMeta(
    loader=partial(
        SiglipModelWrapper,
        model_name="google/siglip-so400m-patch14-384",
    ),
    name="google/siglip-so400m-patch14-384",
    languages=["eng-Latn"],
    revision="9fdffc58afc957d1a03a25b10dba0329ab15c2a3",
    release_date="2024-01-08",
    modalities=["image", "text"],
    n_parameters=878_000_000,
    memory_usage_mb=3349,
    max_tokens=64,
    embed_dim=1152,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/google-research/big_vision/blob/main/big_vision/trainers/proj/image_text/siglip.py",
    public_training_data=None,
    framework=["PyTorch"],
    reference="https://huggingface.co/google/siglip-so400m-patch14-384",
    similarity_fn_name=None,
    use_instructions=False,
    training_datasets=siglip_training_datasets,
)

siglip_so400m_patch16_256_i18n = ModelMeta(
    loader=partial(
        SiglipModelWrapper,
        model_name="google/siglip-so400m-patch16-256-i18n",
    ),
    name="google/siglip-so400m-patch16-256-i18n",
    languages=["eng-Latn"],
    revision="365d321c0cfdea96bc28e3a29787a11a062681a1",
    release_date="2024-01-08",
    modalities=["image", "text"],
    n_parameters=1_130_000_000,
    memory_usage_mb=4306,
    max_tokens=64,
    embed_dim=1152,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/google-research/big_vision/blob/main/big_vision/trainers/proj/image_text/siglip.py",
    public_training_data=None,
    framework=["PyTorch"],
    reference="https://huggingface.co/google/siglip-so400m-patch16-256-i18n",
    similarity_fn_name=None,
    use_instructions=False,
    training_datasets=siglip_training_datasets,
)

siglip_base_patch16_256_multilingual = ModelMeta(
    loader=partial(
        SiglipModelWrapper,
        model_name="google/siglip-base-patch16-256-multilingual",
    ),
    name="google/siglip-base-patch16-256-multilingual",
    languages=["eng-Latn"],
    revision="8952a4eafcde3cb7ab46b1dd629b33f8784ca9c6",
    release_date="2024-01-08",
    modalities=["image", "text"],
    n_parameters=371_000_000,
    memory_usage_mb=1414,
    max_tokens=64,
    embed_dim=768,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/google-research/big_vision/blob/main/big_vision/trainers/proj/image_text/siglip.py",
    public_training_data=None,
    framework=["PyTorch"],
    reference="https://huggingface.co/google/siglip-base-patch16-256-multilingual",
    similarity_fn_name=None,
    use_instructions=False,
    training_datasets=siglip_training_datasets,
)

siglip_base_patch16_256 = ModelMeta(
    loader=partial(
        SiglipModelWrapper,
        model_name="google/siglip-base-patch16-256",
    ),
    name="google/siglip-base-patch16-256",
    languages=["eng-Latn"],
    revision="b078df89e446d623010d890864d4207fe6399f61",
    release_date="2024-01-08",
    modalities=["image", "text"],
    n_parameters=203_000_000,
    memory_usage_mb=775,
    max_tokens=64,
    embed_dim=768,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/google-research/big_vision/blob/main/big_vision/trainers/proj/image_text/siglip.py",
    public_training_data=None,
    framework=["PyTorch"],
    reference="https://huggingface.co/google/siglip-base-patch16-256",
    similarity_fn_name=None,
    use_instructions=False,
    training_datasets=siglip_training_datasets,
)

siglip_base_patch16_512 = ModelMeta(
    loader=partial(
        SiglipModelWrapper,
        model_name="google/siglip-base-patch16-512",
    ),
    name="google/siglip-base-patch16-512",
    languages=["eng-Latn"],
    revision="753a949581523b60257d93e18391e8c27f72eb22",
    release_date="2024-01-08",
    modalities=["image", "text"],
    n_parameters=204_000_000,
    memory_usage_mb=777,
    max_tokens=64,
    embed_dim=768,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/google-research/big_vision/blob/main/big_vision/trainers/proj/image_text/siglip.py",
    public_training_data=None,
    framework=["PyTorch"],
    reference="https://huggingface.co/google/siglip-base-patch16-512",
    similarity_fn_name=None,
    use_instructions=False,
    training_datasets=siglip_training_datasets,
)

siglip_base_patch16_384 = ModelMeta(
    loader=partial(
        SiglipModelWrapper,
        model_name="google/siglip-base-patch16-384",
    ),
    name="google/siglip-base-patch16-384",
    languages=["eng-Latn"],
    revision="41aec1c83b32e0a6fca20ad88ba058aa5b5ea394",
    release_date="2024-01-08",
    modalities=["image", "text"],
    n_parameters=203_000_000,
    memory_usage_mb=776,
    max_tokens=64,
    embed_dim=768,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/google-research/big_vision/blob/main/big_vision/trainers/proj/image_text/siglip.py",
    public_training_data=None,
    framework=["PyTorch"],
    reference="https://huggingface.co/google/siglip-base-patch16-384",
    similarity_fn_name=None,
    use_instructions=False,
    training_datasets=siglip_training_datasets,
)

siglip_base_patch16_224 = ModelMeta(
    loader=partial(
        SiglipModelWrapper,
        model_name="google/siglip-base-patch16-224",
    ),
    name="google/siglip-base-patch16-224",
    languages=["eng-Latn"],
    revision="7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed",
    release_date="2024-01-08",
    modalities=["image", "text"],
    n_parameters=203_000_000,
    memory_usage_mb=775,
    max_tokens=64,
    embed_dim=768,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/google-research/big_vision/blob/main/big_vision/trainers/proj/image_text/siglip.py",
    public_training_data=None,
    framework=["PyTorch"],
    reference="https://huggingface.co/google/siglip-base-patch16-224",
    similarity_fn_name=None,
    use_instructions=False,
    training_datasets=siglip_training_datasets,
)

siglip_large_patch16_256 = ModelMeta(
    loader=partial(
        SiglipModelWrapper,
        model_name="google/siglip-large-patch16-256",
    ),
    name="google/siglip-large-patch16-256",
    languages=["eng-Latn"],
    revision="d0da9f876e7d66b4e250cd2450c3ba2ce735e447",
    release_date="2024-01-08",
    modalities=["image", "text"],
    n_parameters=652_000_000,
    memory_usage_mb=2488,
    max_tokens=64,
    embed_dim=1024,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/google-research/big_vision/blob/main/big_vision/trainers/proj/image_text/siglip.py",
    public_training_data=None,
    framework=["PyTorch"],
    reference="https://huggingface.co/google/siglip-large-patch16-256",
    similarity_fn_name=None,
    use_instructions=False,
    training_datasets=siglip_training_datasets,
)

siglip_large_patch16_384 = ModelMeta(
    loader=partial(
        SiglipModelWrapper,
        model_name="google/siglip-large-patch16-384",
    ),
    name="google/siglip-large-patch16-384",
    languages=["eng-Latn"],
    revision="ce005573a40965dfd21fd937fbdeeebf2439fc35",
    release_date="2024-01-08",
    modalities=["image", "text"],
    n_parameters=652_000_000,
    memory_usage_mb=2489,
    max_tokens=64,
    embed_dim=1024,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/google-research/big_vision/blob/main/big_vision/trainers/proj/image_text/siglip.py",
    public_training_data=None,
    framework=["PyTorch"],
    reference="https://huggingface.co/google/siglip-large-patch16-384",
    similarity_fn_name=None,
    use_instructions=False,
    training_datasets=siglip_training_datasets,
)
