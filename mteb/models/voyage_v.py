from __future__ import annotations

import logging
import os
from functools import partial
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from mteb.encoder_interface import PromptType
from mteb.model_meta import ModelMeta

api_key = os.getenv("VOYAGE_API_KEY")
tensor_to_image = transforms.Compose([transforms.ToPILImage()])


def downsample_image(
    image: Image.Image, max_pixels: int = 16000000, target_longest_side: int = 4000
) -> Image.Image:
    """If image pixel > max_pixels, downsample it to target_longest_side while keeping the width height ratio."""
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


def voyage_v_loader(**kwargs):
    try:
        import voyageai
    except ImportError:
        raise ImportError("To use voyage models, please run `pip install -U voyageai`.")
    try:
        from tenacity import retry, stop_after_attempt, wait_exponential
    except ImportError:
        raise ImportError(
            "please run `pip install tenacity` to use exponential backoff."
        )

    class VoyageMultiModalModelWrapper:
        def __init__(
            self,
            model_name: str,
            **kwargs: Any,
        ):
            self.model_name = model_name
            self.vo = voyageai.Client()

        @retry(
            stop=stop_after_attempt(6),  # Stop after 6 attempts
            wait=wait_exponential(multiplier=1, max=60),  # Exponential backoff
        )
        def _multimodal_embed(self, inputs, model, input_type):
            return self.vo.multimodal_embed(inputs, model=model, input_type=input_type)

        def get_text_embeddings(
            self,
            texts: list[str],
            *,
            task_name: str | None = None,
            prompt_type: PromptType | None = None,
            batch_size: int = 32,
            input_type=None,
            **kwargs: Any,
        ):
            if input_type is None and prompt_type is not None:
                if prompt_type == PromptType.passage:
                    input_type = "document"
                elif prompt_type == PromptType.query:
                    input_type = "query"

            all_text_embeddings = []

            batch_size = 128  # for run tasks purpose

            for i in tqdm(range(0, len(texts), batch_size)):
                batch_texts = texts[i : i + batch_size]
                batch_texts = [[text] for text in batch_texts]

                # with retry mechanism
                embeddings = self._multimodal_embed(
                    batch_texts, model=self.model_name, input_type=input_type
                ).embeddings
                all_text_embeddings.append(torch.tensor(embeddings))
            all_text_embeddings = torch.vstack(all_text_embeddings)
            return all_text_embeddings

        def get_image_embeddings(
            self,
            images: list[Image.Image] | DataLoader,
            *,
            task_name: str | None = None,
            prompt_type: PromptType | None = None,
            batch_size: int = 32,
            input_type=None,
            **kwargs: Any,
        ):
            if input_type is None and prompt_type is not None:
                if prompt_type == PromptType.passage:
                    input_type = "document"
                elif prompt_type == PromptType.query:
                    input_type = "query"

            all_image_embeddings = []

            if isinstance(images, DataLoader):
                for index, batch in enumerate(tqdm(images)):
                    if index == 0:
                        assert len(batch) == batch_size
                    batch_images = [
                        [downsample_image(tensor_to_image(image))] for image in batch
                    ]
                    embeddings = self._multimodal_embed(
                        batch_images, model=self.model_name, input_type=input_type
                    ).embeddings
                    all_image_embeddings.append(torch.tensor(embeddings))
            else:
                for i in tqdm(range(0, len(images), batch_size)):
                    batch_images = images[i : i + batch_size]
                    batch_images = [[downsample_image(image)] for image in batch_images]
                    embeddings = self._multimodal_embed(
                        batch_images, model=self.model_name, input_type=input_type
                    ).embeddings
                    all_image_embeddings.append(torch.tensor(embeddings))
            all_image_embeddings = torch.vstack(all_image_embeddings)
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
            *,
            task_name: str | None = None,
            prompt_type: PromptType | None = None,
            batch_size: int = 32,
            input_type=None,
            **kwargs: Any,
        ):
            if texts is None and images is None:
                raise ValueError("Either texts or images must be provided")

            if input_type is None and prompt_type is not None:
                if prompt_type == PromptType.passage:
                    input_type = "document"
                elif prompt_type == PromptType.query:
                    input_type = "query"

            text_embeddings = None
            image_embeddings = None

            interleaved_embeddings = []
            if texts is not None and images is not None:
                if isinstance(images, DataLoader):
                    for index, batch in tqdm(enumerate(images)):
                        if index == 0:
                            assert len(batch) == batch_size
                        batch_images = [
                            downsample_image(tensor_to_image(image)) for image in batch
                        ]
                        batch_texts = texts[
                            index * batch_size : (index + 1) * batch_size
                        ]
                        interleaved_inputs = [
                            [text, image]
                            for image, text in zip(batch_images, batch_texts)
                        ]
                        embeddings = self._multimodal_embed(
                            interleaved_inputs,
                            model=self.model_name,
                            input_type=input_type,
                        ).embeddings
                        interleaved_embeddings.append(torch.tensor(embeddings))
                else:
                    for i in tqdm(range(0, len(images), batch_size)):
                        batch_images = images[i : i + batch_size]
                        batch_texts = texts[i : i + batch_size]
                        interleaved_inputs = [
                            [text, image]
                            for image, text in zip(batch_images, batch_texts)
                        ]
                        embeddings = self._multimodal_embed(
                            interleaved_inputs,
                            model=self.model_name,
                            input_type=input_type,
                        ).embeddings
                        interleaved_embeddings.append(torch.tensor(embeddings))
                interleaved_embeddings = torch.vstack(interleaved_embeddings)
                return interleaved_embeddings

            elif texts is not None:
                text_embeddings = self.get_text_embeddings(
                    texts, batch_size, input_type=input_type
                )

            elif images is not None:
                image_embeddings = self.get_image_embeddings(
                    images, batch_size, input_type=input_type
                )

            if text_embeddings is not None:
                return text_embeddings
            elif image_embeddings is not None:
                return image_embeddings

    return VoyageMultiModalModelWrapper(**kwargs)


voyage_v = ModelMeta(
    loader=partial(voyage_v_loader, model_name="voyage-multimodal-3"),
    name="voyage-multimodal-3",
    languages=[],  # Unknown
    revision="1",
    release_date="2024-11-10",
    n_parameters=None,
    memory_usage_mb=None,
    max_tokens=None,
    embed_dim=1024,
    license=None,
    similarity_fn_name="cosine",
    framework=[],
    modalities=["image", "text"],
    open_weights=None,
    public_training_code=None,
    public_training_data=None,
    reference=None,
    use_instructions=None,
    training_datasets=None,
)
