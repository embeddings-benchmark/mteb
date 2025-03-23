from __future__ import annotations

from typing import Any, Literal

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

from mteb.abstasks import TaskMetadata
from mteb.model_meta import ModelMeta, ScoringFunction
from mteb.types import Array, BatchedInput, PromptType


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
        self,
        texts: DataLoader[BatchedInput],
        show_progress_bar: bool = True,
        **kwargs: Any,
    ):
        raise ValueError("DINO models only support image encoding.")

    def get_image_embeddings(
        self,
        images: DataLoader[BatchedInput],
        show_progress_bar: bool = True,
        pooling: Literal["cls", "mean"] = "cls",
        **kwargs: Any,
    ):
        all_image_embeddings = []

        with torch.no_grad():
            for batch in tqdm(
                images, disable=not show_progress_bar, desc="Image Encoding"
            ):
                inputs = self.processor(images=batch["image"], return_tensors="pt")
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
            raise ValueError("DINO models only support image encoding.")
        elif text_embeddings is not None:
            return text_embeddings
        elif image_embeddings is not None:
            return image_embeddings
        raise ValueError("No text or image data found.")


dinov2_training_datasets = {
    # LVD-142M
    #  ImageNet-22k
}


dinov2_small = ModelMeta(
    loader=DINOModelWrapper,  # type: ignore
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
    similarity_fn_name=ScoringFunction.VISION,
    use_instructions=False,
    training_datasets=dinov2_training_datasets,
)

dinov2_base = ModelMeta(
    loader=DINOModelWrapper,  # type: ignore
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
    similarity_fn_name=ScoringFunction.VISION,
    use_instructions=False,
    training_datasets=dinov2_training_datasets,
)

dinov2_large = ModelMeta(
    loader=DINOModelWrapper,  # type: ignore
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
    similarity_fn_name=ScoringFunction.VISION,
    use_instructions=False,
    training_datasets=dinov2_training_datasets,
)

dinov2_giant = ModelMeta(
    loader=DINOModelWrapper,  # type: ignore
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
    similarity_fn_name=ScoringFunction.VISION,
    use_instructions=False,
    training_datasets=dinov2_training_datasets,
)
