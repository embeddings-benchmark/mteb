from __future__ import annotations

from functools import partial
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

from mteb.encoder_interface import PromptType
from mteb.model_meta import ModelMeta


class DINOModelWrapper:
    """A wrapper class for DINO models that supports image encoding.
    Text encoding and text-image fusion are not supported.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs: Any,
    ):
        self.model_name = model_name
        self.device = device
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.processor = AutoImageProcessor.from_pretrained(model_name)

    @staticmethod
    def get_text_embeddings(
        texts: list[str],
        *,
        task_name: str | None = None,
        prompt_type: PromptType | None = None,
        batch_size: int = 32,
        **kwargs: Any,
    ):
        raise ValueError("DINO models only support image encoding.")

    def get_image_embeddings(
        self,
        images: list[Image.Image] | DataLoader,
        *,
        task_name: str | None = None,
        prompt_type: PromptType | None = None,
        batch_size: int = 32,
        pooling="cls",
        **kwargs: Any,
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
        raise ValueError("DINO models only support image encoding.")

    def get_fused_embeddings(
        self,
        texts: list[str] = None,
        images: list[Image.Image] | DataLoader = None,
        fusion_mode="sum",
        **kwargs: Any,
    ):
        if texts is None and images is None:
            raise ValueError("images must be provided for DINO models")

        text_embeddings = None
        image_embeddings = None

        if texts is not None:
            text_embeddings = self.get_text_embeddings(texts, **kwargs)

        if images is not None:
            image_embeddings = self.get_image_embeddings(images, **kwargs)

        if text_embeddings is not None and image_embeddings is not None:
            raise ValueError("DINO models only support image encoding.")
        elif text_embeddings is not None:
            return text_embeddings
        elif image_embeddings is not None:
            return image_embeddings


dinov2_training_datasets = {
    # LVD-142M
    #  ImageNet-22k
}


dinov2_small = ModelMeta(
    loader=partial(
        DINOModelWrapper,
        model_name="facebook/dinov2-small",
    ),
    name="facebook/dinov2-small",
    languages=["eng_Latn"],
    revision="ed25f3a31f01632728cabb09d1542f84ab7b0056",
    release_date="2023-07-18",
    modalities=["image"],
    n_parameters=22_100_000,
    memory_usage_mb=84,
    max_tokens=None,
    embed_dim=384,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/facebookresearch/dinov2",
    public_training_data=None,
    framework=["PyTorch"],
    reference="https://huggingface.co/facebook/dinov2-small",
    similarity_fn_name=None,
    use_instructions=False,
    training_datasets=dinov2_training_datasets,
)

dinov2_base = ModelMeta(
    loader=partial(
        DINOModelWrapper,
        model_name="facebook/dinov2-base",
    ),
    name="facebook/dinov2-base",
    languages=["eng_Latn"],
    revision="f9e44c814b77203eaa57a6bdbbd535f21ede1415",
    release_date="2023-07-18",
    modalities=["image"],
    n_parameters=86_600_000,
    memory_usage_mb=330,
    max_tokens=None,
    embed_dim=768,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/facebookresearch/dinov2",
    public_training_data=None,
    framework=["PyTorch"],
    reference="https://huggingface.co/facebook/dinov2-base",
    similarity_fn_name=None,
    use_instructions=False,
    training_datasets=dinov2_training_datasets,
)

dinov2_large = ModelMeta(
    loader=partial(
        DINOModelWrapper,
        model_name="facebook/dinov2-large",
    ),
    name="facebook/dinov2-large",
    languages=["eng_Latn"],
    revision="47b73eefe95e8d44ec3623f8890bd894b6ea2d6c",
    release_date="2023-07-18",
    modalities=["image"],
    n_parameters=304_000_000,
    memory_usage_mb=1161,
    max_tokens=None,
    embed_dim=1024,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/facebookresearch/dinov2",
    public_training_data=None,
    framework=["PyTorch"],
    reference="https://huggingface.co/facebook/dinov2-large",
    similarity_fn_name=None,
    use_instructions=False,
    training_datasets=dinov2_training_datasets,
)

dinov2_giant = ModelMeta(
    loader=partial(
        DINOModelWrapper,
        model_name="facebook/dinov2-giant",
    ),
    name="facebook/dinov2-giant",
    languages=["eng_Latn"],
    revision="611a9d42f2335e0f921f1e313ad3c1b7178d206d",
    release_date="2023-07-18",
    modalities=["image"],
    n_parameters=1_140_000_000,
    memory_usage_mb=4335,
    max_tokens=None,
    embed_dim=1536,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/facebookresearch/dinov2",
    public_training_data=None,
    framework=["PyTorch"],
    reference="https://huggingface.co/facebook/dinov2-giant",
    similarity_fn_name=None,
    use_instructions=False,
    training_datasets=dinov2_training_datasets,
)
