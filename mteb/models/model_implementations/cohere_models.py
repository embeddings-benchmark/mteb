import logging
import time
from functools import wraps
from typing import Any, Literal, get_args

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from mteb._requires_package import requires_package
from mteb.abstasks.task_metadata import TaskMetadata
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.types import Array, BatchedInput, PromptType

logger = logging.getLogger(__name__)

supported_languages = [
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

# Cohere API limits
COHERE_MAX_BATCH_SIZE = 96  # Maximum number of texts per API call
COHERE_MAX_TOKENS_PER_BATCH = 128_000  # Maximum total tokens per API call


def retry_with_rate_limit(
    max_retries: int = 5,
    max_rpm: int = 300,
    initial_delay: float = 1.0,
):
    """Combined retry and rate limiting decorator.

    This decorator handles both proactive rate limiting (spacing requests)
    and reactive retry with exponential backoff for API errors.

    The decorator will use instance attributes (self.max_retries, self.max_rpm)
    if they exist, otherwise falls back to the decorator parameters.

    Args:
        max_retries: Default maximum number of retry attempts (default: 5)
        max_rpm: Default maximum requests per minute for rate limiting (default: 300)
        initial_delay: Initial delay in seconds for exponential backoff (default: 1.0)
    """
    previous_call_ts: float | None = None

    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            import cohere

            nonlocal previous_call_ts

            request_interval = 60.0 / max_rpm

            # Rate limiting: wait before making request if needed
            current_time = time.time()
            if (
                previous_call_ts is not None
                and current_time - previous_call_ts < request_interval
            ):
                time.sleep(request_interval - (current_time - previous_call_ts))

            # Retry logic with exponential backoff
            for attempt in range(max_retries):
                try:
                    result = func(self, *args, **kwargs)
                    previous_call_ts = time.time()
                    return result
                except cohere.errors.TooManyRequestsError as e:
                    if attempt == max_retries - 1:
                        raise
                    # For rate limits, wait longer (30s minimum to respect API limits)
                    delay = max(30, initial_delay * (2**attempt))
                    logger.warning(
                        f"Cohere rate limit (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {delay}s..."
                    )
                    time.sleep(delay)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    delay = initial_delay * (2**attempt)
                    logger.warning(
                        f"Cohere API error (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {delay}s..."
                    )
                    time.sleep(delay)

        return wrapper

    return decorator


# Implementation follows https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/blob/main/src/seb/registered_models/cohere_models.py
class CohereTextEmbeddingModel(AbsEncoder):
    def __init__(
        self,
        model_name: str,
        revision: str,
        sep: str = " ",
        model_prompts: dict[str, str] | None = None,
        embedding_type: EmbeddingType = "float",
        output_dimension: int | None = None,
        **kwargs,
    ) -> None:
        requires_package(self, "cohere", model_name, "pip install 'mteb[cohere]'")

        import cohere

        self.model_name = model_name.removeprefix("Cohere/Cohere-")
        self.sep = sep
        self.model_prompts = self.validate_task_to_prompt_name(model_prompts)
        if embedding_type not in get_args(EmbeddingType):
            raise ValueError(
                f"Embedding type {embedding_type} not allowed. Choose from {get_args(EmbeddingType)}"
            )
        self.embedding_type = embedding_type
        self.output_dimension = output_dimension

        self._client = cohere.Client()

    @retry_with_rate_limit(max_retries=5, max_rpm=300)
    def _embed_func(self, **kwargs):
        """Call Cohere embed API with retry and rate limiting."""
        return self._client.embed(**kwargs)

    def _embed(
        self,
        sentences: list[str],
        cohere_task_type: str,
        show_progress_bar: bool = False,
    ) -> torch.Tensor:
        all_embeddings = []
        index = 0

        pbar = tqdm(
            total=len(sentences),
            desc="Encoding sentences",
            disable=not show_progress_bar,
        )

        while index < len(sentences):
            # Build batch respecting both count and token limits
            batch, batch_tokens = [], 0
            while (
                index < len(sentences)
                and len(batch) < COHERE_MAX_BATCH_SIZE
                and batch_tokens < COHERE_MAX_TOKENS_PER_BATCH
            ):
                # Count tokens for current sentence
                n_tokens = len(
                    self._client.tokenize(
                        text=sentences[index], model=self.model_name
                    ).tokens
                )

                # Check if adding this sentence would exceed token limit
                if (
                    batch_tokens + n_tokens > COHERE_MAX_TOKENS_PER_BATCH
                    and len(batch) > 0
                ):
                    break

                batch_tokens += n_tokens
                batch.append(sentences[index])
                index += 1

            # Embed the batch with retry logic handled by client
            embed_kwargs = {
                "texts": batch,
                "model": self.model_name,
                "input_type": cohere_task_type,
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
                raise ValueError(f"Embedding type {self.embedding_type} not allowed")

            all_embeddings.extend(torch.tensor(embeddings).numpy())
            pbar.update(len(batch))

        pbar.close()
        embeddings_array = np.array(all_embeddings)

        # Post-process embeddings based on type (similar to voyage_models.py)
        primary_embedding_type = self.embedding_type

        if primary_embedding_type == "binary":
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
            embeddings_array = np.array(unpacked_embeddings, dtype=np.float32)
        elif primary_embedding_type in ["int8", "uint8"]:
            # Convert int8/uint8 embeddings to float32
            embeddings_array = embeddings_array.astype(np.float32)

        return embeddings_array

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
        prompt_name = self.get_prompt_name(task_metadata, prompt_type)
        cohere_task_type = self.model_prompts.get(prompt_name)

        if cohere_task_type is None:
            # search_document is recommended if unknown (https://cohere.com/blog/introducing-embed-v3)
            cohere_task_type = "search_document"

        show_progress_bar = (
            False
            if "show_progress_bar" not in kwargs
            else kwargs.pop("show_progress_bar")
        )
        inputs = [text for batch in inputs for text in batch["text"]]

        return self._embed(
            inputs,
            cohere_task_type=cohere_task_type,
            show_progress_bar=show_progress_bar,
        )


model_prompts = {
    "Classification": "classification",
    "MultilabelClassification": "classification",
    "Clustering": "clustering",
    PromptType.query.value: "search_query",
    PromptType.document.value: "search_document",
}

cohere_mult_3 = ModelMeta(
    loader=CohereTextEmbeddingModel,
    loader_kwargs=dict(
        model_prompts=model_prompts,
    ),
    name="Cohere/Cohere-embed-multilingual-v3.0",
    model_type=["dense"],
    languages=supported_languages,
    open_weights=False,
    revision="1",
    release_date="2023-11-02",
    n_parameters=None,
    memory_usage_mb=None,
    max_tokens=None,
    embed_dim=512,
    reference="https://cohere.com/blog/introducing-embed-v3",
    license=None,
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["API"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,  # assumed
    training_datasets=None,
)

cohere_eng_3 = ModelMeta(
    loader=CohereTextEmbeddingModel,
    loader_kwargs=dict(
        model_prompts=model_prompts,
    ),
    name="Cohere/Cohere-embed-english-v3.0",
    model_type=["dense"],
    languages=["eng-Latn"],
    open_weights=False,
    reference="https://cohere.com/blog/introducing-embed-v3",
    revision="1",
    release_date="2023-11-02",
    n_parameters=None,
    memory_usage_mb=None,
    max_tokens=512,
    embed_dim=1024,
    license=None,
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["API"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,  # assumed
    training_datasets=None,
)

cohere_mult_light_3 = ModelMeta(
    loader=CohereTextEmbeddingModel,
    loader_kwargs=dict(
        model_prompts=model_prompts,
    ),
    name="Cohere/Cohere-embed-multilingual-light-v3.0",
    model_type=["dense"],
    languages=supported_languages,
    open_weights=False,
    revision="1",
    reference="https://cohere.com/blog/introducing-embed-v3",
    release_date="2023-11-02",
    n_parameters=None,
    memory_usage_mb=None,
    max_tokens=512,
    embed_dim=384,
    license=None,
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["API"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,  # assumed
    training_datasets=None,
)

cohere_eng_light_3 = ModelMeta(
    loader=CohereTextEmbeddingModel,
    loader_kwargs=dict(
        model_prompts=model_prompts,
    ),
    name="Cohere/Cohere-embed-english-light-v3.0",
    model_type=["dense"],
    languages=["eng-Latn"],
    open_weights=False,
    reference="https://cohere.com/blog/introducing-embed-v3",
    revision="1",
    release_date="2023-11-02",
    n_parameters=None,
    memory_usage_mb=None,
    max_tokens=512,
    embed_dim=384,
    license=None,
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["API"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,  # assumed
    training_datasets=None,
)
