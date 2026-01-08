from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from mteb._requires_package import requires_image_dependencies, requires_package
from mteb.abstasks.task_metadata import TaskMetadata
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.types import Array, BatchedInput, PromptType

if TYPE_CHECKING:
    from PIL import Image


def _downsample_image(
    image: Image.Image, max_pixels: int = 16000000, target_longest_side: int = 4000
) -> Image.Image:
    """If image pixel > max_pixels, downsample it to target_longest_side while keeping the width height ratio.

    Returns:
        The downsampled image.
    """
    width, height = image.size
    pixels = width * height

    if pixels > max_pixels:
        if width > height:
            new_width = target_longest_side
            new_height = int(height * (target_longest_side / width))
        else:
            new_height = target_longest_side
            new_width = int(width * (target_longest_side / height))

        new_size = (new_width, new_height)
        logging.info(
            f"Downsampling image from {width}x{height} to {new_width}x{new_height}"
        )
        return image.resize(new_size, Image.LANCZOS)
    if width > height:
        if width > 10000:
            logging.error("Processing extremely wide images.")
            return image.resize((10000, height), Image.LANCZOS)
    else:
        if height > 10000:
            logging.error("Processing extremely high images.")
            return image.resize((width, 10000), Image.LANCZOS)
    return image


def voyage_v_loader(model_name, **kwargs):
    requires_package(
        voyage_v_loader,
        "voyageai",
        model_name,
        "pip install 'mteb[voyage_v]'",
    )
    requires_package(
        voyage_v_loader,
        "tenacity",
        model_name,
        "pip install 'mteb[voyage_v]'",
    )
    import voyageai
    from tenacity import retry, stop_after_attempt, wait_exponential

    class VoyageMultiModalModelWrapper(AbsEncoder):
        def __init__(
            self,
            model_name: str,
            **kwargs: Any,
        ):
            requires_image_dependencies()

            self.model_name = model_name.split("/")[-1]
            self.vo = voyageai.Client()

        @retry(
            stop=stop_after_attempt(6),  # Stop after 6 attempts
            wait=wait_exponential(multiplier=1, max=60),  # Exponential backoff
        )
        def _multimodal_embed(self, inputs, model, input_type):
            return self.vo.multimodal_embed(inputs, model=model, input_type=input_type)

        def get_text_embeddings(
            self,
            texts: DataLoader[BatchedInput],
            show_progress_bar: bool = True,
            prompt_type: PromptType | None = None,
            input_type: Literal["document", "query"] | None = None,
            **kwargs: Any,
        ):
            if input_type is None and prompt_type is not None:
                if prompt_type == PromptType.document:
                    input_type = "document"
                elif prompt_type == PromptType.query:
                    input_type = "query"

            all_text_embeddings = []

            for batch in tqdm(
                texts, disable=not show_progress_bar, desc="Text Encoding"
            ):
                batch_texts = [[text] for text in batch["text"]]

                # with retry mechanism
                embeddings = self._multimodal_embed(
                    batch_texts, model=self.model_name, input_type=input_type
                ).embeddings
                all_text_embeddings.append(torch.tensor(embeddings))
            all_text_embeddings = torch.vstack(all_text_embeddings)
            return all_text_embeddings

        def get_image_embeddings(
            self,
            images: DataLoader[BatchedInput],
            show_progress_bar: bool = True,
            prompt_type: PromptType | None = None,
            input_type: Literal["document", "query"] | None = None,
            **kwargs: Any,
        ):
            if input_type is None and prompt_type is not None:
                if prompt_type == PromptType.document:
                    input_type = "document"
                elif prompt_type == PromptType.query:
                    input_type = "query"

            all_image_embeddings = []

            for batch in tqdm(
                images, disable=not show_progress_bar, desc="Image Encoding"
            ):
                batch_images = [[_downsample_image(image)] for image in batch["image"]]
                embeddings = self._multimodal_embed(
                    batch_images, model=self.model_name, input_type=input_type
                ).embeddings
                all_image_embeddings.append(torch.tensor(embeddings))
            all_image_embeddings = torch.vstack(all_image_embeddings)
            return all_image_embeddings

        def encode(
            self,
            inputs: DataLoader[BatchedInput],
            *,
            task_metadata: TaskMetadata,
            hf_split: str,
            hf_subset: str,
            prompt_type: PromptType | None = None,
            show_progress_bar: bool = True,
            **kwargs: Any,
        ) -> Array:
            input_type = "document"  # default
            if prompt_type is not None:
                if prompt_type == PromptType.document:
                    input_type = "document"
                elif prompt_type == PromptType.query:
                    input_type = "query"

            text_embeddings = None
            image_embeddings = None

            interleaved_embeddings = []
            if "text" in inputs.dataset.features and "image" in inputs.dataset.features:
                for batch in tqdm(
                    inputs, disable=not show_progress_bar, desc="Interleaved Encoding"
                ):
                    batch_images = [
                        _downsample_image(image) for image in batch["image"]
                    ]
                    batch_texts = batch["text"]
                    interleaved_inputs = [
                        [text, image] for image, text in zip(batch_images, batch_texts)
                    ]
                    embeddings = self._multimodal_embed(
                        interleaved_inputs,
                        model=self.model_name,
                        input_type=input_type,
                    ).embeddings
                    interleaved_embeddings.append(torch.tensor(embeddings))
                interleaved_embeddings = torch.vstack(interleaved_embeddings)
                return interleaved_embeddings
            elif "text" in inputs.dataset.features:
                text_embeddings = self.get_text_embeddings(
                    inputs, input_type=input_type
                )
            elif "image" in inputs.dataset.features:
                image_embeddings = self.get_image_embeddings(
                    inputs, input_type=input_type
                )

            if text_embeddings is not None:
                return text_embeddings
            elif image_embeddings is not None:
                return image_embeddings
            raise ValueError

    return VoyageMultiModalModelWrapper(model_name, **kwargs)


voyage_v = ModelMeta(
    loader=voyage_v_loader,
    name="voyageai/voyage-multimodal-3",
    model_type=["dense"],
    languages=[],  # Unknown
    revision="1",
    release_date="2024-11-10",
    n_parameters=None,
    memory_usage_mb=None,
    max_tokens=32768,
    embed_dim=1024,
    license="mit",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["API"],
    modalities=["image", "text"],
    open_weights=False,
    public_training_code=None,
    public_training_data=None,
    reference="https://huggingface.co/voyageai/voyage-multimodal-3",
    use_instructions=None,
    training_datasets=set(),  # No overlap with MTEB according to Voyage, could overlap with MIEB, didn't ask
)
