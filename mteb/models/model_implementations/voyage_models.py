import time
from functools import wraps
from typing import Any, Literal

import numpy as np
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from mteb._requires_package import requires_package
from mteb.abstasks.task_metadata import TaskMetadata
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.types import Array, BatchedInput, PromptType

VOYAGE_TRAINING_DATA = set(
    # Self-reported (message from VoyageAI member)
    # synthetic data
)

# The missing values are translated to themselves
VOYAGE_DTYPE_TRANSLATION = {
    "float32": "float",
    "bf16": "float",
}

# Total token limits per model based on VoyageAI documentation
VOYAGE_TOTAL_TOKEN_LIMITS = {
    "voyage-3.5-lite": 1_000_000,
    "voyage-3.5": 320_000,
    "voyage-2": 320_000,
    "voyage-3-large": 120_000,
    "voyage-code-3": 120_000,
    "voyage-large-2-instruct": 120_000,
    "voyage-finance-2": 120_000,
    "voyage-multilingual-2": 120_000,
    "voyage-law-2": 120_000,
    "voyage-large-2": 120_000,
    "voyage-3": 120_000,
    "voyage-3-lite": 120_000,
    "voyage-code-2": 120_000,
    "voyage-3-m-exp": 120_000,
}


def token_limit(max_tpm: int, interval: int = 60):
    limit_interval_start_ts = time.time()
    used_tokens = 0

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal limit_interval_start_ts, used_tokens

            result = func(*args, **kwargs)
            used_tokens += result.total_tokens

            current_time = time.time()
            if current_time - limit_interval_start_ts > interval:
                limit_interval_start_ts = current_time
                used_tokens = 0

            if used_tokens > max_tpm:
                time.sleep(interval - (current_time - limit_interval_start_ts))
                used_tokens = 0
            return result

        return wrapper

    return decorator


def rate_limit(max_rpm: int, interval: int = 60):
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


class VoyageModel(AbsEncoder):
    def __init__(
        self,
        model_name: str,
        max_retries: int = 5,
        max_rpm: int = 300,
        max_tpm: int = 1_000_000,
        max_tokens: int | None = None,
        model_prompts: dict[str, str] | None = None,
        output_dtype: str | None = None,
        **kwargs,
    ) -> None:
        requires_package(self, "voyageai", model_name, "pip install 'mteb[voyageai]'")
        import voyageai

        self._client = voyageai.Client(max_retries=max_retries)
        self._embed_func = rate_limit(max_rpm)(token_limit(max_tpm)(self._client.embed))

        self._model_name = model_name.split("/")[-1].split()[0]
        self._max_tpm = max_tpm
        self._max_tokens = max_tokens
        self.model_prompts = self.validate_task_to_prompt_name(model_prompts)
        self.output_dtype = output_dtype
        self._max_tokens_per_batch = VOYAGE_TOTAL_TOKEN_LIMITS.get(
            self._model_name, 120_000
        )

    def encode(
        self,
        inputs: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        batch_size: int = 1_000,  # https://docs.voyageai.com/reference/embeddings-api
        **kwargs: Any,
    ) -> Array:
        prompt_name = self.get_prompt_name(task_metadata, prompt_type)
        input_type = self.model_prompts.get(prompt_name, "document")
        sentences = [text for batch in inputs for text in batch["text"]]
        return self._batched_encode(sentences, batch_size, input_type)

    def _batched_encode(
        self,
        sentences: list[str],
        batch_size: int,
        input_type: Literal["query", "document"],
    ) -> np.ndarray:
        embeddings, index = [], 0

        output_dtype = VOYAGE_DTYPE_TRANSLATION.get(
            self.output_dtype, self.output_dtype
        )

        pbar = tqdm(total=len(sentences), desc="Encoding sentences")
        while index < len(sentences):
            batch, batch_tokens = [], 0
            while (
                index < len(sentences)
                and len(batch) < batch_size
                and batch_tokens < self._max_tokens_per_batch
            ):
                txt = sentences[index] if len(sentences[index]) > 0 else " "
                n_tokens = len(self._client.tokenize([txt], model=self._model_name)[0])
                if (
                    batch_tokens + n_tokens > self._max_tokens_per_batch
                    and len(batch) > 0
                ):
                    break
                batch_tokens += n_tokens
                batch.append(txt)
                index += 1

            embeddings.extend(
                self._embed_func(
                    texts=batch,
                    model=self._model_name,
                    input_type=input_type,
                    output_dtype=output_dtype,
                ).embeddings
            )
            pbar.update(len(batch))

        pbar.close()
        embeddings_array = np.array(embeddings)

        if output_dtype == "binary":
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
        elif output_dtype != "float":
            # Convert int8/uint8 embeddings to float32
            embeddings_array = embeddings_array.astype(np.float32)

        return embeddings_array


model_prompts = {
    PromptType.query.value: "query",
    PromptType.document.value: "document",
}

voyage_3_large = ModelMeta(
    name="voyageai/voyage-3-large",  # Date of publication of this post https://blog.voyageai.com/2025/01/07/voyage-3-large/
    revision="1",
    release_date="2025-01-07",
    languages=None,  # supported languages not specified
    loader=VoyageModel,
    loader_kwargs=dict(
        max_tokens=32000,
        model_prompts=model_prompts,
    ),
    max_tokens=32000,
    embed_dim=1024,
    open_weights=False,
    n_parameters=None,
    memory_usage_mb=None,
    license=None,
    reference="https://blog.voyageai.com/2025/01/07/voyage-3-large/",
    similarity_fn_name="cosine",
    framework=["API"],
    use_instructions=True,
    training_datasets=VOYAGE_TRAINING_DATA,
    public_training_code=None,
    public_training_data=None,
)


voyage_3_5 = ModelMeta(
    name="voyageai/voyage-3.5",
    revision="1",
    release_date="2025-01-21",
    languages=None,  # supported languages not specified
    loader=VoyageModel,
    loader_kwargs=dict(
        max_tokens=32000,
        model_prompts=model_prompts,
    ),
    max_tokens=32000,
    embed_dim=1024,
    open_weights=False,
    n_parameters=None,
    memory_usage_mb=None,
    license=None,
    reference="https://blog.voyageai.com/2025/05/20/voyage-3-5/",
    similarity_fn_name="cosine",
    framework=["API"],
    use_instructions=True,
    training_datasets=VOYAGE_TRAINING_DATA,
    public_training_code=None,
    public_training_data=None,
)

voyage_3_5_int8 = ModelMeta(
    name="voyageai/voyage-3.5 (output_dtype=int8)",
    revision="1",
    release_date="2025-01-21",
    languages=None,  # supported languages not specified
    loader=VoyageModel,
    loader_kwargs=dict(
        model_prompts=model_prompts,
        output_dtype="int8",
    ),
    max_tokens=32000,
    embed_dim=1024,
    open_weights=False,
    n_parameters=None,
    memory_usage_mb=None,
    license=None,
    reference="https://blog.voyageai.com/2025/05/20/voyage-3-5/",
    similarity_fn_name="cosine",
    framework=["API"],
    use_instructions=True,
    training_datasets=VOYAGE_TRAINING_DATA,
    public_training_code=None,
    public_training_data=None,
    adapted_from="voyageai/voyage-3.5",
)

voyage_3_5_binary = ModelMeta(
    name="voyageai/voyage-3.5 (output_dtype=binary)",
    revision="1",
    release_date="2025-01-21",
    languages=None,  # supported languages not specified
    loader=VoyageModel,
    loader_kwargs=dict(
        model_prompts=model_prompts,
        output_dtype="binary",
    ),
    max_tokens=32000,
    embed_dim=1024,  # Same as original after unpacking from bits
    open_weights=False,
    n_parameters=None,
    memory_usage_mb=None,
    license=None,
    reference="https://blog.voyageai.com/2025/05/20/voyage-3-5/",
    similarity_fn_name="cosine",
    framework=["API"],
    use_instructions=True,
    training_datasets=VOYAGE_TRAINING_DATA,
    public_training_code=None,
    public_training_data=None,
    adapted_from="voyageai/voyage-3.5",
)

voyage_large_2_instruct = ModelMeta(
    name="voyageai/voyage-large-2-instruct",
    revision="1",
    release_date="2024-05-05",
    languages=None,  # supported languages not specified
    loader=VoyageModel,
    loader_kwargs=dict(
        model_prompts=model_prompts,
        max_tokens=16000,
    ),
    max_tokens=16000,
    embed_dim=1024,
    open_weights=False,
    n_parameters=None,
    memory_usage_mb=None,
    license=None,
    reference="https://blog.voyageai.com/2024/05/05/voyage-large-2-instruct-instruction-tuned-and-rank-1-on-mteb/",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["API"],
    use_instructions=True,
    training_datasets=VOYAGE_TRAINING_DATA,
    public_training_code=None,
    public_training_data=None,
)

voyage_finance_2 = ModelMeta(
    name="voyageai/voyage-finance-2",
    revision="1",
    release_date="2024-05-30",
    languages=None,  # supported languages not specified
    loader=VoyageModel,
    loader_kwargs=dict(
        model_prompts=model_prompts,
        max_tokens=32000,
    ),
    max_tokens=32000,
    embed_dim=1024,
    open_weights=False,
    n_parameters=None,
    memory_usage_mb=None,
    license=None,
    reference="https://blog.voyageai.com/2024/06/03/domain-specific-embeddings-finance-edition-voyage-finance-2/",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["API"],
    use_instructions=True,
    training_datasets=VOYAGE_TRAINING_DATA,
    public_training_code=None,
    public_training_data=None,
)

voyage_law_2 = ModelMeta(
    name="voyageai/voyage-law-2",
    revision="1",
    release_date="2024-04-15",
    languages=None,  # supported languages not specified
    loader=VoyageModel,
    loader_kwargs=dict(
        model_prompts=model_prompts,
        max_tokens=16000,
    ),
    max_tokens=16000,
    embed_dim=1024,
    open_weights=False,
    n_parameters=None,
    memory_usage_mb=None,
    license=None,
    reference="https://blog.voyageai.com/2024/04/15/domain-specific-embeddings-and-retrieval-legal-edition-voyage-law-2/",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["API"],
    use_instructions=True,
    training_datasets=VOYAGE_TRAINING_DATA,
    public_training_code=None,
    public_training_data=None,
)

voyage_code_2 = ModelMeta(
    name="voyageai/voyage-code-2",
    revision="1",
    release_date="2024-01-23",
    languages=None,  # supported languages not specified
    loader=VoyageModel,
    loader_kwargs=dict(
        model_prompts=model_prompts,
        max_tokens=16000,
    ),
    max_tokens=16000,
    embed_dim=1536,
    open_weights=False,
    n_parameters=None,
    memory_usage_mb=None,
    license=None,
    reference="https://blog.voyageai.com/2024/01/23/voyage-code-2-elevate-your-code-retrieval/",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["API"],
    use_instructions=True,
    training_datasets=VOYAGE_TRAINING_DATA,
    public_training_code=None,
    public_training_data=None,
)

voyage_code_3 = ModelMeta(
    name="voyageai/voyage-code-3",
    revision="1",
    release_date="2024-12-04",
    languages=None,  # supported languages not specified
    loader=VoyageModel,
    loader_kwargs=dict(
        model_prompts=model_prompts,
        max_tokens=32000,
    ),
    max_tokens=32000,
    embed_dim=1024,
    open_weights=False,
    n_parameters=None,
    memory_usage_mb=None,
    license=None,
    reference="https://blog.voyageai.com/2024/12/04/voyage-code-3/",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["API"],
    use_instructions=True,
    training_datasets=VOYAGE_TRAINING_DATA,  # src: private communication with Voyage
    public_training_code=None,
    public_training_data=None,
)


voyage_large_2 = ModelMeta(
    name="voyageai/voyage-large-2",  # Date of publication of this post https://blog.voyageai.com/2023/10/29/voyage-embeddings/
    revision="1",
    release_date="2023-10-29",
    languages=None,  # supported languages not specified
    loader=VoyageModel,
    loader_kwargs=dict(
        model_prompts=model_prompts,
        max_tokens=16000,
    ),
    max_tokens=16000,
    embed_dim=1536,
    open_weights=False,
    n_parameters=None,
    memory_usage_mb=None,
    license=None,
    reference="https://blog.voyageai.com/2023/10/29/voyage-embeddings/",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["API"],
    use_instructions=True,
    training_datasets=VOYAGE_TRAINING_DATA,
    public_training_code=None,
    public_training_data=None,
)

voyage_2 = ModelMeta(
    name="voyageai/voyage-2",
    revision="1",
    release_date="2023-10-29",
    languages=None,  # supported languages not specified
    loader=VoyageModel,
    loader_kwargs=dict(
        model_prompts=model_prompts,
        max_tokens=4000,
    ),
    max_tokens=4000,
    embed_dim=1024,
    open_weights=False,
    n_parameters=None,
    memory_usage_mb=None,
    license=None,
    reference="https://blog.voyageai.com/2023/10/29/voyage-embeddings/",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["API"],
    use_instructions=True,
    training_datasets=VOYAGE_TRAINING_DATA,
    public_training_code=None,
    public_training_data=None,
)
voyage_multilingual_2 = ModelMeta(
    name="voyageai/voyage-multilingual-2",
    revision="1",
    release_date="2024-06-10",
    languages=None,  # supported languages not specified
    loader=VoyageModel,
    loader_kwargs=dict(
        model_prompts=model_prompts,
        max_tokens=32000,
    ),
    max_tokens=32000,
    embed_dim=1024,
    open_weights=False,
    n_parameters=None,
    memory_usage_mb=None,
    license=None,
    reference="https://blog.voyageai.com/2024/06/10/voyage-multilingual-2-multilingual-embedding-model/",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["API"],
    use_instructions=True,
    training_datasets=VOYAGE_TRAINING_DATA,
    public_training_code=None,
    public_training_data=None,
)

voyage_3 = ModelMeta(
    name="voyageai/voyage-3",
    revision="1",
    release_date="2024-09-18",
    languages=None,  # supported languages not specified
    loader=VoyageModel,
    loader_kwargs=dict(
        model_prompts=model_prompts,
        max_tokens=32000,
    ),
    max_tokens=32000,
    embed_dim=1024,
    open_weights=False,
    n_parameters=None,
    memory_usage_mb=None,
    license=None,
    reference="https://blog.voyageai.com/2024/09/18/voyage-3/",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["API"],
    use_instructions=True,
    training_datasets=VOYAGE_TRAINING_DATA,
    public_training_code=None,
    public_training_data=None,
)

voyage_3_lite = ModelMeta(
    name="voyageai/voyage-3-lite",
    revision="1",
    release_date="2024-09-18",
    languages=None,  # supported languages not specified
    loader=VoyageModel,
    loader_kwargs=dict(
        model_prompts=model_prompts,
        max_tokens=32000,
    ),
    max_tokens=32000,
    embed_dim=512,
    open_weights=False,
    n_parameters=None,
    memory_usage_mb=None,
    license=None,
    reference="https://blog.voyageai.com/2024/09/18/voyage-3/",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["API"],
    use_instructions=True,
    training_datasets=VOYAGE_TRAINING_DATA,
    public_training_code=None,
    public_training_data=None,
)

voyage_3_exp = ModelMeta(
    name="voyageai/voyage-3-m-exp",
    revision="1",
    release_date="2025-01-08",
    languages=["eng-Latn"],
    loader=VoyageModel,
    loader_kwargs=dict(
        model_prompts=model_prompts,
        max_tokens=32000,
    ),
    max_tokens=32000,
    embed_dim=2048,
    open_weights=False,
    # from their card https://huggingface.co/voyageai/voyage-3-m-exp#model-information
    n_parameters=int(6918 * 1e6),
    memory_usage_mb=None,
    license=None,
    reference="https://huggingface.co/voyageai/voyage-3-m-exp",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["API"],
    use_instructions=True,
    training_datasets={
        # MTEB(eng, v1) training data:
        "AmazonPolarityClassification",
        "AmazonReviewsClassification",
        "EmotionClassification",
        "HotpotQA",
        "ImdbClassification",
        "MTOPDomainClassification",
        "MTOPIntentClassification",
        "MindSmallReranking",
        "MassiveIntentClassification",
        "MassiveScenarioClassification",
        "MedrxivClusteringP2P",
        "MedrxivClusteringS2S",
        "STS12",
        "STSBenchmark",
        "StackOverflowDupQuestions",
        "ToxicConversationsClassification",
        "TweetSentimentExtractionClassification",
        "BiorxivClusteringP2P",
        "BiorxivClusteringS2S",
        "Banking77Classification",
        "ArguAna",
        "ArguAna-PL",
        "ArguAna-NL",  # translation not trained on
        "NanoArguAnaRetrieval",
        "STS22",
        "AmazonCounterfactualClassification",
        "ArxivClusteringP2P",
        "ArxivClusteringS2S",
        "NQ",
        "SciFact",
        "QuoraRetrieval",
        "NanoQuoraRetrieval",
        "NQHardNegatives",
        "NanoNQRetrieval",
        "NQ-PL",  # translation not trained on
        "NQ-NL",  # translation not trained on
        "NFCorpus",
        "FEVERHardNegatives",
        "NanoFEVERRetrieval",
        "FEVER-NL",  # translation not trained on
        "FiQA2018-NL",  # translation not trained on
        "BiorxivClusteringP2P.v2",
        "BiorxivClusteringS2S.v2",
        "MedrxivClusteringP2P.v2",
        "MedrxivClusteringS2S.v2",
        "MSMARCO",
        "MSMARCOHardNegatives",
        "NanoMSMARCORetrieval",
        "MSMARCO-PL",  # translation not trained on
        "mMARCO-NL",  # translation not trained on
        "HotpotQA-PL",  # translation not trained on
        "HotpotQA-NL",  # translation not trained on
        "HotpotQAHardNegatives",
        "FEVER",
        "FiQA2018",
        "DBPedia",
        "TRECCOVID",
        "ArxivClusteringP2P.v2",
        "STSBenchmarkMultilingualSTS",  # translated, not trained on
    },
    public_training_code=None,
    public_training_data=None,
)
