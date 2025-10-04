from __future__ import annotations

import base64
import io
import os
import time
from functools import partial, wraps
from typing import Any, Literal, get_args

import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from mteb.encoder_interface import PromptType
from mteb.model_meta import ModelMeta
from mteb.requires_package import requires_image_dependencies, requires_package


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

EMBEDDING_TYPE = Literal[
    "float",
    "int8",
    "uint8",
    "binary",
]

# Cohere API limits
COHERE_MAX_BATCH_SIZE = 96  # Maximum number of texts per API call
COHERE_MAX_TOKENS_PER_BATCH = 128_000  # Maximum total tokens per API call

# Per-model context lengths (max tokens per individual input)
COHERE_MODEL_CONTEXT_LENGTHS = {
    "embed-english-v3.0": 512,
    "embed-multilingual-v3.0": 512,
    "embed-english-light-v3.0": 512,
    "embed-multilingual-light-v3.0": 512,
    "embed-v4.0": 128_000,
}


def rate_limit(max_rpm: int, interval: int = 60):
    """Rate limiter decorator to respect requests per minute limit."""
    request_interval = interval / max_rpm
    previous_call_ts: float | None = None

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_time = time.time()
            nonlocal previous_call_ts
            if (
                previous_call_ts is not None
                and current_time - previous_call_ts < request_interval
            ):
                time.sleep(request_interval - (current_time - previous_call_ts))

            result = func(*args, **kwargs)
            previous_call_ts = time.time()
            return result

        return wrapper

    return decorator


def retry_with_exponential_backoff(max_retries: int = 5, initial_delay: float = 1.0):
    """Retry decorator with exponential backoff for handling any errors.

    Retries on any exception with exponential backoff.
    Rate limit errors (429) wait a minimum of 30 seconds.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            import cohere

            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except cohere.errors.TooManyRequestsError as e:
                    if attempt == max_retries - 1:
                        raise
                    # For rate limits, wait longer (30s minimum to respect API limits)
                    delay = max(30, initial_delay * (2**attempt))
                    print(
                        f"Cohere rate limit (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {delay}s..."
                    )
                    time.sleep(delay)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    delay = initial_delay * (2**attempt)
                    print(
                        f"Cohere API error (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {delay}s..."
                    )
                    time.sleep(delay)

        return wrapper

    return decorator


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
            embedding_type: EMBEDDING_TYPE = "float",
            output_dimension: int | None = None,
            max_retries: int = 5,
            max_rpm: int = 300,
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
            assert embedding_type in get_args(EMBEDDING_TYPE)
            self.embedding_type = embedding_type
            self.output_dimension = output_dimension
            api_key = os.getenv("COHERE_API_KEY")

            # Create Cohere client with retry and rate limiting
            self.client = cohere.ClientV2(api_key)
            self._embed_func = retry_with_exponential_backoff(max_retries)(
                rate_limit(max_rpm)(self.client.embed)
            )
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
            index = 0

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
                            # Fallback for unknown types
                            embeddings = response.embeddings.float
                        all_image_embeddings.append(torch.tensor(embeddings))
                        time.sleep(1.5)
            all_image_embeddings = torch.cat(all_image_embeddings, dim=0)

            # Post-process embeddings based on type
            all_image_embeddings = _post_process_embeddings(
                all_image_embeddings, self.embedding_type
            )

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

cohere_embed_v4_multimodal = ModelMeta(
    loader=partial(cohere_v_loader, model_name="embed-v4.0"),
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
    loader=partial(cohere_v_loader, model_name="embed-v4.0", embedding_type="binary"),
    name="Cohere/Cohere-embed-v4.0 (output_dtype=binary)",
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
    loader=partial(cohere_v_loader, model_name="embed-v4.0", embedding_type="int8"),
    name="Cohere/Cohere-embed-v4.0 (output_dtype=int8)",
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
