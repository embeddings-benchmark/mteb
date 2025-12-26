import base64
import io
import os
import time
from typing import Any, Literal, get_args

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from mteb._requires_package import requires_image_dependencies, requires_package
from mteb.abstasks.task_metadata import TaskMetadata
from mteb.models import ModelMeta
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_implementations.cohere_models import (
    COHERE_MAX_BATCH_SIZE,
    COHERE_MAX_TOKENS_PER_BATCH,
    retry_with_rate_limit,
)
from mteb.models.model_meta import ScoringFunction
from mteb.types import Array, BatchedInput, PromptType


def _post_process_embeddings(
    embeddings_array: torch.Tensor, embedding_type: str
) -> torch.Tensor:
    """Post-process embeddings based on type (similar to voyage_models.py)"""
    if embedding_type == "binary":
        # Unpack bit-packed embeddings: each byte contains 8 embedding values
        unpacked_embeddings = []
        for embedding in embeddings_array:
            # Convert bytes to bits and unpack
            unpacked = []
            for byte_val in embedding:
                # Extract 8 bits from each byte (LSB first)
                for bit_pos in range(8):
                    bit_val = (byte_val >> bit_pos) & 1
                    # Convert 0/1 to -1/1 for binary (signed)
                    unpacked.append(1.0 if bit_val else -1.0)
            unpacked_embeddings.append(unpacked)
        return torch.tensor(unpacked_embeddings, dtype=torch.float32)
    elif embedding_type in ["int8", "uint8"]:
        # Convert int8/uint8 embeddings to float32
        return embeddings_array.float()
    else:
        # For float and other types, return as-is
        return embeddings_array


all_languages = [
    "afr-Latn",
    "amh-Ethi",
    "ara-Arab",
    "asm-Beng",
    "aze-Latn",
    "bel-Cyrl",
    "bul-Cyrl",
    "ben-Beng",
    "bod-Tibt",
    "bos-Latn",
    "cat-Latn",
    "ceb-Latn",
    "cos-Latn",
    "ces-Latn",
    "cym-Latn",
    "dan-Latn",
    "deu-Latn",
    "ell-Grek",
    "eng-Latn",
    "epo-Latn",
    "spa-Latn",
    "est-Latn",
    "eus-Latn",
    "fas-Arab",
    "fin-Latn",
    "fra-Latn",
    "fry-Latn",
    "gle-Latn",
    "gla-Latn",
    "glg-Latn",
    "guj-Gujr",
    "hau-Latn",
    "haw-Latn",
    "heb-Hebr",
    "hin-Deva",
    "hmn-Latn",
    "hrv-Latn",
    "hat-Latn",
    "hun-Latn",
    "hye-Armn",
    "ind-Latn",
    "ibo-Latn",
    "isl-Latn",
    "ita-Latn",
    "jpn-Jpan",
    "jav-Latn",
    "kat-Geor",
    "kaz-Cyrl",
    "khm-Khmr",
    "kan-Knda",
    "kor-Kore",
    "kur-Arab",
    "kir-Cyrl",
    "lat-Latn",
    "ltz-Latn",
    "lao-Laoo",
    "lit-Latn",
    "lav-Latn",
    "mlg-Latn",
    "mri-Latn",
    "mkd-Cyrl",
    "mal-Mlym",
    "mon-Cyrl",
    "mar-Deva",
    "msa-Latn",
    "mlt-Latn",
    "mya-Mymr",
    "nep-Deva",
    "nld-Latn",
    "nor-Latn",
    "nob-Latn",
    "nno-Latn",
    "nya-Latn",
    "ori-Orya",
    "pan-Guru",
    "pol-Latn",
    "por-Latn",
    "ron-Latn",
    "rus-Cyrl",
    "kin-Latn",
    "sin-Sinh",
    "slk-Latn",
    "slv-Latn",
    "smo-Latn",
    "sna-Latn",
    "som-Latn",
    "sqi-Latn",
    "srp-Cyrl",
    "sot-Latn",
    "sun-Latn",
    "swe-Latn",
    "swa-Latn",
    "tam-Taml",
    "tel-Telu",
    "tgk-Cyrl",
    "tha-Thai",
    "tuk-Latn",
    "tgl-Latn",
    "tur-Latn",
    "tat-Cyrl",
    "uig-Arab",
    "ukr-Cyrl",
    "urd-Arab",
    "uzb-Latn",
    "vie-Latn",
    "wol-Latn",
    "xho-Latn",
    "yid-Hebr",
    "yor-Latn",
    "zho-Hans",
    "zul-Latn",
]

EmbeddingType = Literal[
    "float",
    "int8",
    "uint8",
    "binary",
]


def cohere_v_loader(model_name, **kwargs):
    requires_package(
        cohere_v_loader, "cohere", model_name, "pip install 'mteb[cohere]'"
    )
    import cohere

    class CohereMultiModalModelWrapper(AbsEncoder):
        def __init__(
            self,
            model_name: str,
            embedding_type: EmbeddingType = "float",
            output_dimension: int | None = None,
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
            if embedding_type not in get_args(EmbeddingType):
                raise ValueError(
                    f"Embedding type {embedding_type} not allowed. Choose from {get_args(EmbeddingType)}"
                )
            self.embedding_type = embedding_type
            self.output_dimension = output_dimension
            api_key = os.getenv("COHERE_API_KEY")
            self.client = cohere.ClientV2(api_key)
            self.image_format = "JPEG"
            self.transform = transforms.Compose([transforms.PILToTensor()])

        @retry_with_rate_limit(max_retries=5, max_rpm=300)
        def _embed_func(self, **kwargs):
            """Call Cohere embed API with retry and rate limiting."""
            return self.client.embed(**kwargs)

        def get_text_embeddings(
            self,
            inputs: DataLoader[BatchedInput],
            show_progress_bar: bool = True,
            **kwargs: Any,
        ):
            all_text_embeddings = []
            index = 0
            texts = [text for batch in inputs for text in batch["text"]]

            pbar = tqdm(total=len(texts), desc="Encoding text sentences")

            while index < len(texts):
                # Build batch respecting both count and token limits
                batch, batch_tokens = [], 0
                while (
                    index < len(texts)
                    and len(batch) < COHERE_MAX_BATCH_SIZE
                    and batch_tokens < COHERE_MAX_TOKENS_PER_BATCH
                ):
                    # Count tokens for current sentence
                    n_tokens = len(
                        self.client.tokenize(
                            text=texts[index], model=self.model_name
                        ).tokens
                    )

                    # Check if adding this sentence would exceed token limit
                    if (
                        batch_tokens + n_tokens > COHERE_MAX_TOKENS_PER_BATCH
                        and len(batch) > 0
                    ):
                        break

                    batch_tokens += n_tokens
                    batch.append(texts[index])
                    index += 1

                # Embed the batch with retry logic handled by client
            for batch in tqdm(
                texts, disable=not show_progress_bar, desc="Text Encoding"
            ):
                embed_kwargs = {
                    "texts": batch,
                    "model": self.model_name,
                    "input_type": "search_document",
                    "embedding_types": [self.embedding_type],
                }
                if self.output_dimension is not None:
                    embed_kwargs["output_dimension"] = self.output_dimension

                response = self._embed_func(**embed_kwargs)

                # Get embeddings based on requested type
                if self.embedding_type == "float":
                    embeddings = response.embeddings.float
                elif self.embedding_type == "int8":
                    embeddings = response.embeddings.int8
                elif self.embedding_type == "uint8":
                    embeddings = response.embeddings.uint8
                elif self.embedding_type == "binary":
                    embeddings = response.embeddings.binary
                else:
                    raise ValueError(
                        f"Embedding type {self.embedding_type} not allowed"
                    )
                all_text_embeddings.append(torch.tensor(embeddings))
                pbar.update(len(batch))

            pbar.close()
            all_text_embeddings = torch.cat(all_text_embeddings, dim=0)

            # Post-process embeddings based on type
            all_text_embeddings = _post_process_embeddings(
                all_text_embeddings, self.embedding_type
            )

            return all_text_embeddings

        def get_image_embeddings(
            self,
            images: DataLoader[BatchedInput],
            show_progress_bar: bool = True,
            **kwargs: Any,
        ):
            all_image_embeddings = []
            images = [image for batch in images for image in batch["images"]]

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
                    embed_kwargs = {
                        "model": self.model_name,
                        "input_type": "image",
                        "embedding_types": [self.embedding_type],
                        "images": [image_base64],
                    }
                    if self.output_dimension is not None:
                        embed_kwargs["output_dimension"] = self.output_dimension

                        response = self._embed_func(**embed_kwargs)

                        # Get embeddings based on requested type
                        if self.embedding_type == "float":
                            embeddings = response.embeddings.float
                        elif self.embedding_type == "int8":
                            embeddings = response.embeddings.int8
                        elif self.embedding_type == "uint8":
                            embeddings = response.embeddings.uint8
                        elif self.embedding_type == "binary":
                            embeddings = response.embeddings.binary
                        else:
                            raise ValueError(
                                f"Embedding type {self.embedding_type} not allowed"
                            )
                        all_image_embeddings.append(torch.tensor(embeddings))
                        time.sleep(1.5)
            all_image_embeddings = torch.cat(all_image_embeddings, dim=0)

            # Post-process embeddings based on type
            all_image_embeddings = _post_process_embeddings(
                all_image_embeddings, self.embedding_type
            )

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

    return CohereMultiModalModelWrapper(model_name, **kwargs)


cohere_mult_3 = ModelMeta(
    loader=cohere_v_loader,
    loader_kwargs={"model_name": "embed-multilingual-v3.0"},
    name="cohere/embed-multilingual-v3.0",
    model_type=["dense"],
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
    loader=cohere_v_loader,
    loader_kwargs={"model_name": "embed-english-v3.0"},
    name="cohere/embed-english-v3.0",
    model_type=["dense"],
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

cohere_embed_v4_multimodal = ModelMeta(
    loader=cohere_v_loader,
    loader_kwargs=dict(model_name="embed-v4.0"),
    model_type=["dense"],
    name="Cohere/Cohere-embed-v4.0",
    languages=all_languages,
    revision="1",
    release_date="2024-12-01",
    n_parameters=None,
    memory_usage_mb=None,
    max_tokens=128000,
    embed_dim=1536,
    license=None,
    similarity_fn_name="cosine",
    framework=[],
    modalities=["image", "text"],
    open_weights=False,
    public_training_code=None,
    public_training_data=None,
    reference="https://docs.cohere.com/docs/cohere-embed",
    use_instructions=False,
    training_datasets=None,
)

cohere_embed_v4_multimodal_binary = ModelMeta(
    loader=cohere_v_loader,
    loader_kwargs=dict(embedding_type="binary"),
    name="Cohere/Cohere-embed-v4.0 (output_dtype=binary)",
    model_type=["dense"],
    languages=all_languages,
    revision="1",
    release_date="2024-12-01",
    n_parameters=None,
    memory_usage_mb=None,
    max_tokens=128000,
    embed_dim=1536,
    license=None,
    similarity_fn_name="cosine",
    framework=[],
    modalities=["image", "text"],
    open_weights=False,
    public_training_code=None,
    public_training_data=None,
    reference="https://docs.cohere.com/docs/embeddings",
    use_instructions=False,
    training_datasets=None,
    adapted_from="Cohere/Cohere-embed-v4.0",
)

cohere_embed_v4_multimodal_int8 = ModelMeta(
    loader=cohere_v_loader,
    loader_kwargs=dict(embedding_type="int8"),
    name="Cohere/Cohere-embed-v4.0 (output_dtype=int8)",
    model_type=["dense"],
    languages=all_languages,
    revision="1",
    release_date="2024-12-01",
    n_parameters=None,
    memory_usage_mb=None,
    max_tokens=128000,
    embed_dim=1536,
    license=None,
    similarity_fn_name="cosine",
    framework=[],
    modalities=["image", "text"],
    open_weights=False,
    public_training_code=None,
    public_training_data=None,
    reference="https://docs.cohere.com/docs/embeddings",
    use_instructions=False,
    training_datasets=None,
    adapted_from="Cohere/Cohere-embed-v4.0",
)
