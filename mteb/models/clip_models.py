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


class CLIPModelWrapper:
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
                    text=batch_texts, return_tensors="pt", padding=True, truncation=True
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
                    inputs = self.processor(
                        images=batch_images, return_tensors="pt", padding=True
                    )
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    image_outputs = self.model.get_image_features(**inputs)
                    all_image_embeddings.append(image_outputs.cpu())

        all_image_embeddings = torch.cat(all_image_embeddings, dim=0)
        return all_image_embeddings

    def calculate_probs(self, text_embeddings, image_embeddings):
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
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
                raise ValueError(f"fusion mode {fusion_mode} hasn't been implemented")
            return fused_embeddings
        elif text_embeddings is not None:
            return text_embeddings
        elif image_embeddings is not None:
            return image_embeddings


clip_vit_large_patch14 = ModelMeta(
    loader=partial(
        CLIPModelWrapper,
        model_name="openai/clip-vit-large-patch14",
    ),
    name="openai/clip-vit-large-patch14",
    languages=["eng_Latn"],
    revision="32bd64288804d66eefd0ccbe215aa642df71cc41",
    release_date="2021-02-26",
    modalities=["image", "text"],
    n_parameters=428_000_000,
    memory_usage_mb=1631,
    max_tokens=77,
    embed_dim=768,
    license=None,
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["PyTorch"],
    reference="https://huggingface.co/openai/clip-vit-large-patch14",
    similarity_fn_name=None,
    use_instructions=False,
    training_datasets=None,
)

clip_vit_base_patch32 = ModelMeta(
    loader=partial(
        CLIPModelWrapper,
        model_name="openai/clip-vit-base-patch32",
    ),
    name="openai/clip-vit-base-patch32",
    languages=["eng_Latn"],
    revision="3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268",
    release_date="2021-02-26",
    modalities=["image", "text"],
    n_parameters=151_000_000,
    memory_usage_mb=576,
    max_tokens=77,
    embed_dim=512,
    license=None,
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["PyTorch"],
    reference="https://huggingface.co/openai/clip-vit-base-patch32",
    similarity_fn_name=None,
    use_instructions=False,
    training_datasets=None,
)

clip_vit_base_patch16 = ModelMeta(
    loader=partial(
        CLIPModelWrapper,
        model_name="openai/clip-vit-base-patch16",
    ),
    name="openai/clip-vit-base-patch16",
    languages=["eng_Latn"],
    revision="57c216476eefef5ab752ec549e440a49ae4ae5f3",
    release_date="2021-02-26",
    modalities=["image", "text"],
    n_parameters=151_000_000,
    memory_usage_mb=576,
    max_tokens=77,
    embed_dim=512,
    license=None,
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["PyTorch"],
    reference="https://huggingface.co/openai/clip-vit-base-patch16",
    similarity_fn_name=None,
    use_instructions=False,
    training_datasets=None,
)
