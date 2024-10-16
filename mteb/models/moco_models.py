from __future__ import annotations

from functools import partial
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import ViTModel, AutoProcessor

from mteb.model_meta import ModelMeta


class MOCOv3Wrapper:
    """A wrapper class for MOCOv3 models that supports image encoding.
    Text encoding and text-image fusion are not supported.
    """
    def __init__(
        self,
        model_name: str = "nyu-visionx/moco-v3-vit-b",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs: Any,
    ):
        self.model_name = model_name
        self.device = device
        self.model = ViTModel.from_pretrained(model_name, trust_remote_code=True).to(
            self.device
        )
        self.processor = AutoProcessor.from_pretrained("laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K")

    @staticmethod
    def get_text_embeddings(texts: list[str], batch_size: int = 32):
        raise ValueError("MOCO models only support image encoding.")
    
    def get_image_embeddings(
        self,
        images: list[Image.Image] | DataLoader,
        batch_size: int = 32,
        pooling="cls",
    ):
        all_image_embeddings = []

        if isinstance(images, DataLoader):
            with torch.no_grad():
                for batch in tqdm(images):
                    inputs = self.processor(images=batch, return_tensors="pt")
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    image_outputs = self.model(**inputs)
                    features = image_outputs.last_hidden_state
                    if pooling == "cls":
                        features = features[:, 0, :]  # TODO: confirm best practice
                    elif pooling == "mean":
                        features = features.mean(dim=1)
                    else:
                        raise ValueError(
                            "Pooling methods not implemented. Use cls or mean."
                        )
                    all_image_embeddings.append(features.cpu())
        else:
            with torch.no_grad():
                for i in tqdm(range(0, len(images), batch_size)):
                    batch_images = images[i : i + batch_size]
                    inputs = self.processor(images=batch_images, return_tensors="pt")
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    image_outputs = self.model(**inputs)
                    features = image_outputs.last_hidden_state
                    if pooling == "cls":
                        features = features[:, 0, :]
                    elif pooling == "mean":
                        features = features.mean(dim=1)
                    else:
                        raise ValueError(
                            "Pooling methods not implemented. Use cls or mean."
                        )
                    all_image_embeddings.append(features.cpu())

        all_image_embeddings = torch.cat(all_image_embeddings, dim=0)
        return all_image_embeddings

    @staticmethod
    def calculate_probs(text_embeddings, image_embeddings):
        raise ValueError("MOCO models only support image encoding.")

    def get_fused_embeddings(
        self,
        texts: list[str] = None,
        images: list[Image.Image] | DataLoader = None,
        fusion_mode="sum",
        batch_size: int = 32,
    ):
        if texts is None and images is None:
            raise ValueError("images must be provided for MOCO models")

        text_embeddings = None
        image_embeddings = None

        if texts is not None:
            text_embeddings = self.get_text_embeddings(texts, batch_size)

        if images is not None:
            image_embeddings = self.get_image_embeddings(images, batch_size)

        if text_embeddings is not None and image_embeddings is not None:
            raise ValueError("MOCO models only support image encoding.")
        elif text_embeddings is not None:
            return text_embeddings
        elif image_embeddings is not None:
            return image_embeddings

mocov3_vit_base = ModelMeta(
    loader=partial(
        MOCOv3Wrapper,
        model_name="nyu-visionx/moco-v3-vit-b",
    ),
    name="nyu-visionx/moco-v3-vit-b",
    languages=["eng_Latn"],
    open_source=True,
    revision="7d091cd70772c5c0ecf7f00b5f12ca609a99d69d",
    release_date="2024-06-03",
)

mocov3_vit_large = ModelMeta(
    loader=partial(
        MOCOv3Wrapper,
        model_name="nyu-visionx/moco-v3-vit-l",
    ),
    name="nyu-visionx/moco-v3-vit-l",
    languages=["eng_Latn"],
    open_source=True,
    revision="7bf75358d616f39b9716148bf4e3425f3bd35b47",
    release_date="2024-06-03",
)
