from __future__ import annotations

import base64
import io
import os
import time
from functools import partial
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from mteb.encoder_interface import PromptType
from mteb.model_meta import ModelMeta
from mteb.requires_package import requires_image_dependencies, requires_package


def cohere_v_loader(**kwargs):
    model_name = kwargs.get("model_name", "Cohere")
    requires_package(
        cohere_v_loader, "cohere", model_name, "pip install 'mteb[cohere]'"
    )
    import cohere

    class CohereMultiModalModelWrapper:
        def __init__(
            self,
            model_name: str,
            **kwargs: Any,
        ):
            """Wrapper for Cohere multimodal embedding model,

            do `export COHERE_API_KEY=<Your_Cohere_API_KEY>` before running eval scripts.
            Cohere currently supports 40 images/min, thus time.sleep(1.5) is applied after each image.
            Remove or adjust this after Cohere API changes capacity.
            """
            requires_image_dependencies()
            from torchvision import transforms

            self.model_name = model_name
            api_key = os.getenv("COHERE_API_KEY")
            self.client = cohere.ClientV2(api_key)
            self.image_format = "JPEG"
            self.transform = transforms.Compose([transforms.PILToTensor()])

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

            for i in tqdm(range(0, len(texts), batch_size)):
                batch_texts = texts[i : i + batch_size]
                response = self.client.embed(
                    texts=batch_texts,
                    model=self.model_name,
                    input_type="search_document",
                )
                all_text_embeddings.append(torch.tensor(response.embeddings.float))

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
                for batch in tqdm(images):
                    for image in batch:
                        # cohere only supports 1 image per call
                        buffered = io.BytesIO()
                        image = self.transform(image)
                        image.save(buffered, format=self.image_format)
                        image_bytes = buffered.getvalue()
                        stringified_buffer = base64.b64encode(image_bytes).decode(
                            "utf-8"
                        )
                        content_type = f"image/{self.image_format.lower()}"
                        image_base64 = (
                            f"data:{content_type};base64,{stringified_buffer}"
                        )
                        response = self.client.embed(
                            model=self.model_name,
                            input_type="image",
                            embedding_types=["float"],
                            images=[image_base64],
                        )
                        all_image_embeddings.append(
                            torch.tensor(response.embeddings.float)
                        )
                        time.sleep(1.5)
            else:
                for i in tqdm(range(0, len(images), batch_size)):
                    batch_images = images[i : i + batch_size]
                    for image in batch_images:
                        # cohere only supports 1 image per call
                        buffered = io.BytesIO()
                        image.save(buffered, format=self.image_format)
                        image_bytes = buffered.getvalue()
                        stringified_buffer = base64.b64encode(image_bytes).decode(
                            "utf-8"
                        )
                        content_type = f"image/{self.image_format.lower()}"
                        image_base64 = (
                            f"data:{content_type};base64,{stringified_buffer}"
                        )
                        response = self.client.embed(
                            model=self.model_name,
                            input_type="image",
                            embedding_types=["float"],
                            images=[image_base64],
                        )
                        all_image_embeddings.append(
                            torch.tensor(response.embeddings.float)
                        )
                        time.sleep(1.5)
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
            texts: list[str] | None = None,
            images: list[Image.Image] | DataLoader | None = None,
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

    return CohereMultiModalModelWrapper(**kwargs)


cohere_mult_3 = ModelMeta(
    loader=partial(cohere_v_loader, model_name="embed-multilingual-v3.0"),
    name="Cohere/Cohere-embed-multilingual-v3.0",
    languages=[],  # Unknown, but support >100 languages
    revision="1",
    release_date="2024-10-24",
    n_parameters=None,
    memory_usage_mb=None,
    max_tokens=None,
    embed_dim=1024,
    license=None,
    similarity_fn_name="cosine",
    framework=[],
    modalities=["image", "text"],
    open_weights=False,
    public_training_code=None,
    public_training_data=None,
    reference="https://huggingface.co/Cohere/Cohere-embed-multilingual-v3.0",
    use_instructions=False,
    training_datasets=None,
)

cohere_eng_3 = ModelMeta(
    loader=partial(cohere_v_loader, model_name="embed-english-v3.0"),
    name="Cohere/Cohere-embed-english-v3.0",
    languages=["eng-Latn"],
    revision="1",
    release_date="2024-10-24",
    n_parameters=None,
    memory_usage_mb=None,
    max_tokens=None,
    embed_dim=1024,
    license=None,
    similarity_fn_name="cosine",
    framework=[],
    modalities=["image", "text"],
    open_weights=False,
    public_training_code=None,
    public_training_data=None,
    reference="https://huggingface.co/Cohere/Cohere-embed-english-v3.0",
    use_instructions=False,
    training_datasets=None,
)
