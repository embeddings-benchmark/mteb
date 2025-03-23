from __future__ import annotations

import base64
import io
import os
import time
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from mteb.abstasks import TaskMetadata
from mteb.model_meta import ModelMeta, ScoringFunction
from mteb.requires_package import requires_image_dependencies
from mteb.types import Array, BatchedInput, PromptType


def cohere_v_loader(**kwargs):
    try:
        import cohere  # type: ignore
    except ImportError:
        raise ImportError("To use cohere models, please run `pip install cohere`.")

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
            texts: DataLoader[BatchedInput],
            show_progress_bar: bool = True,
            **kwargs: Any,
        ):
            all_text_embeddings = []

            for batch in tqdm(
                texts, disable=not show_progress_bar, desc="Text Encoding"
            ):
                response = self.client.embed(
                    texts=batch["text"],
                    model=self.model_name,
                    input_type="search_document",
                )
                all_text_embeddings.append(torch.tensor(response.embeddings.float))

            all_text_embeddings = torch.cat(all_text_embeddings, dim=0)
            return all_text_embeddings

        def get_image_embeddings(
            self,
            images: DataLoader[BatchedInput],
            show_progress_bar: bool = True,
            **kwargs: Any,
        ):
            all_image_embeddings = []

            for batch in tqdm(
                images, disable=not show_progress_bar, desc="Image Encoding"
            ):
                for image in batch:
                    # cohere only supports 1 image per call
                    buffered = io.BytesIO()
                    image = self.transform(image)
                    image.save(buffered, format=self.image_format)
                    image_bytes = buffered.getvalue()
                    stringified_buffer = base64.b64encode(image_bytes).decode("utf-8")
                    content_type = f"image/{self.image_format.lower()}"
                    image_base64 = f"data:{content_type};base64,{stringified_buffer}"
                    response = self.client.embed(
                        model=self.model_name,
                        input_type="image",
                        embedding_types=["float"],
                        images=[image_base64],
                    )
                    all_image_embeddings.append(torch.tensor(response.embeddings.float))
                    time.sleep(1.5)
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

    return CohereMultiModalModelWrapper(**kwargs)


cohere_mult_3 = ModelMeta(
    loader=cohere_v_loader,  # type: ignore
    loader_kwargs={"model_name": "embed-multilingual-v3.0"},
    name="cohere/embed-multilingual-v3.0",
    languages=[],  # Unknown, but support >100 languages
    revision="1",
    release_date="2024-10-24",
    n_parameters=None,
    memory_usage_mb=None,
    max_tokens=None,
    embed_dim=1024,
    license=None,
    similarity_fn_name=ScoringFunction.COSINE,
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
    loader=cohere_v_loader,  # type: ignore
    loader_kwargs={"model_name": "embed-english-v3.0"},
    name="cohere/embed-english-v3.0",
    languages=["eng-Latn"],
    revision="1",
    release_date="2024-10-24",
    n_parameters=None,
    memory_usage_mb=None,
    max_tokens=None,
    embed_dim=1024,
    license=None,
    similarity_fn_name=ScoringFunction.COSINE,
    framework=[],
    modalities=["image", "text"],
    open_weights=False,
    public_training_code=None,
    public_training_data=None,
    reference="https://huggingface.co/Cohere/Cohere-embed-english-v3.0",
    use_instructions=False,
    training_datasets=None,
)
