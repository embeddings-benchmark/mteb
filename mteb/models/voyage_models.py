from __future__ import annotations

import time
from functools import partial, wraps
from typing import Any, Literal

import numpy as np

from mteb.encoder_interface import PromptType
from mteb.model_meta import ModelMeta
from mteb.models.wrapper import Wrapper
from mteb.requires_package import requires_package

VOYAGE_TRAINING_DATA = {
    # Self-reported (message from VoyageAI member)
    # synthetic data
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


class VoyageWrapper(Wrapper):
    def __init__(
        self,
        model_name: str,
        max_retries: int = 5,
        max_rpm: int = 300,
        max_tpm: int = 1_000_000,
        model_prompts: dict[str, str] | None = None,
        **kwargs,
    ) -> None:
        requires_package(self, "voyageai", model_name, "pip install 'mteb[voyageai]'")
        import voyageai

        self._client = voyageai.Client(max_retries=max_retries)
        self._embed_func = rate_limit(max_rpm)(token_limit(max_tpm)(self._client.embed))
        self._model_name = model_name
        self._max_tpm = max_tpm
        self.model_prompts = (
            self.validate_task_to_prompt_name(model_prompts) if model_prompts else None
        )

    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int = 32,
        task_name: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        prompt_name = self.get_prompt_name(self.model_prompts, task_name, prompt_type)
        input_type = self.model_prompts.get(prompt_name, "document")

        return self._batched_encode(sentences, batch_size, input_type)

    def _batched_encode(
        self,
        sentences: list[str],
        batch_size: int,
        input_type: Literal["query", "document"],
    ) -> np.ndarray:
        embeddings, index = [], 0

        while index < len(sentences):
            batch, batch_tokens = [], 0
            while (
                index < len(sentences)
                and len(batch) < batch_size
                and batch_tokens < self._max_tpm
            ):
                n_tokens = len(
                    self._client.tokenize([sentences[index]], model=self._model_name)[0]
                )
                if batch_tokens + n_tokens > self._max_tpm:
                    break
                batch_tokens += n_tokens
                batch.append(sentences[index])
                index += 1

            embeddings.extend(
                self._embed_func(
                    texts=batch,
                    model=self._model_name,
                    input_type=input_type,
                ).embeddings
            )

        return np.array(embeddings)


model_prompts = {
    PromptType.query.value: "query",
    PromptType.passage.value: "document",
}

voyage_large_2_instruct = ModelMeta(
    name="voyageai/voyage-large-2-instruct",
    revision="1",
    release_date="2024-05-05",
    languages=None,  # supported languages not specified
    loader=partial(  # type: ignore
        VoyageWrapper,
        model_name="voyage-large-2-instruct",
        model_prompts=model_prompts,
    ),
    max_tokens=16000,
    embed_dim=1024,
    open_weights=False,
    n_parameters=None,
    memory_usage_mb=None,
    license=None,
    reference="https://blog.voyageai.com/2024/05/05/voyage-large-2-instruct-instruction-tuned-and-rank-1-on-mteb/",
    similarity_fn_name="cosine",
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
    loader=partial(  # type: ignore
        VoyageWrapper,
        model_name="voyage-finance-2",
        model_prompts=model_prompts,
    ),
    max_tokens=32000,
    embed_dim=1024,
    open_weights=False,
    n_parameters=None,
    memory_usage_mb=None,
    license=None,
    reference="https://blog.voyageai.com/2024/06/03/domain-specific-embeddings-finance-edition-voyage-finance-2/",
    similarity_fn_name="cosine",
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
    loader=partial(  # type: ignore
        VoyageWrapper,
        model_name="voyage-law-2",
        model_prompts=model_prompts,
    ),
    max_tokens=16000,
    embed_dim=1024,
    open_weights=False,
    n_parameters=None,
    memory_usage_mb=None,
    license=None,
    reference="https://blog.voyageai.com/2024/04/15/domain-specific-embeddings-and-retrieval-legal-edition-voyage-law-2/",
    similarity_fn_name="cosine",
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
    loader=partial(  # type: ignore
        VoyageWrapper,
        model_name="voyage-code-2",
        model_prompts=model_prompts,
    ),
    max_tokens=16000,
    embed_dim=1536,
    open_weights=False,
    n_parameters=None,
    memory_usage_mb=None,
    license=None,
    reference="https://blog.voyageai.com/2024/01/23/voyage-code-2-elevate-your-code-retrieval/",
    similarity_fn_name="cosine",
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
    loader=partial(  # type: ignore
        VoyageWrapper,
        model_name="voyage-code-3",
        model_prompts=model_prompts,
    ),
    max_tokens=32000,
    embed_dim=1024,
    open_weights=False,
    n_parameters=None,
    memory_usage_mb=None,
    license=None,
    reference="https://blog.voyageai.com/2024/12/04/voyage-code-3/",
    similarity_fn_name="cosine",
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
    loader=partial(  # type: ignore
        VoyageWrapper,
        model_name="voyage-large-2",
        model_prompts=model_prompts,
    ),
    max_tokens=16000,
    embed_dim=1536,
    open_weights=False,
    n_parameters=None,
    memory_usage_mb=None,
    license=None,
    reference="https://blog.voyageai.com/2023/10/29/voyage-embeddings/",
    similarity_fn_name="cosine",
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
    loader=partial(  # type: ignore
        VoyageWrapper,
        model_name="voyage-2",
        model_prompts=model_prompts,
    ),
    max_tokens=4000,
    embed_dim=1024,
    open_weights=False,
    n_parameters=None,
    memory_usage_mb=None,
    license=None,
    reference="https://blog.voyageai.com/2023/10/29/voyage-embeddings/",
    similarity_fn_name="cosine",
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
    loader=partial(  # type: ignore
        VoyageWrapper,
        model_name="voyage-multilingual-2",
        model_prompts=model_prompts,
    ),
    max_tokens=32000,
    embed_dim=1024,
    open_weights=False,
    n_parameters=None,
    memory_usage_mb=None,
    license=None,
    reference="https://blog.voyageai.com/2024/06/10/voyage-multilingual-2-multilingual-embedding-model/",
    similarity_fn_name="cosine",
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
    loader=partial(
        VoyageWrapper,
        model_name="voyage-3",
        model_prompts=model_prompts,
    ),
    max_tokens=32000,
    embed_dim=1024,
    open_weights=False,
    n_parameters=None,
    memory_usage_mb=None,
    license=None,
    reference="https://blog.voyageai.com/2024/09/18/voyage-3/",
    similarity_fn_name="cosine",
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
    loader=partial(
        VoyageWrapper,
        model_name="voyage-3-lite",
        model_prompts=model_prompts,
    ),
    max_tokens=32000,
    embed_dim=512,
    open_weights=False,
    n_parameters=None,
    memory_usage_mb=None,
    license=None,
    reference="https://blog.voyageai.com/2024/09/18/voyage-3/",
    similarity_fn_name="cosine",
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
    loader=partial(
        VoyageWrapper,
        model_name="voyage-3-m-exp",
        model_prompts=model_prompts,
    ),
    max_tokens=32000,
    embed_dim=2048,
    open_weights=False,
    # from their card https://huggingface.co/voyageai/voyage-3-m-exp#model-information
    n_parameters=int(6918 * 1e6),
    memory_usage_mb=None,
    license=None,
    reference="https://huggingface.co/voyageai/voyage-3-m-exp",
    similarity_fn_name="cosine",
    framework=["API"],
    use_instructions=True,
    training_datasets={
        # MTEB(eng, v1) training data:
        "AmazonPolarityClassification": ["train"],
        "AmazonReviewsClassification": ["train"],
        "EmotionClassification": ["train"],
        "HotpotQA": ["train"],
        "ImdbClassification": ["train"],
        "MTOPDomainClassification": ["train"],
        "MTOPIntentClassification": ["train"],
        "MindSmallReranking": ["train"],
        "MassiveIntentClassification": ["train"],
        "MassiveScenarioClassification": ["train"],
        "MedrxivClusteringP2P": ["train"],
        "MedrxivClusteringS2S": ["train"],
        "STS12": ["train"],
        "STSBenchmark": ["train"],
        "StackOverflowDupQuestions": ["train"],
        "ToxicConversationsClassification": ["train"],
        "TweetSentimentExtractionClassification": ["train"],
        "BiorxivClusteringP2P": ["train"],
        "BiorxivClusteringS2S": ["train"],
        "Banking77Classification": ["train"],
        "ArguAna": ["train"],
        "ArguAna-PL": ["train"],
        "ArguAna-NL": ["train"],  # translation not trained on
        "NanoArguAnaRetrieval": ["train"],
        "STS22": ["train"],
        "AmazonCounterfactualClassification": ["train"],
        "ArxivClusteringP2P": ["train"],
        "ArxivClusteringS2S": ["train"],
        "NQ": ["train"],
        "SciFact": ["train"],
        "QuoraRetrieval": ["train"],
        "NanoQuoraRetrieval": ["train"],
        "NQHardNegatives": ["train"],
        "NanoNQRetrieval": ["train"],
        "NQ-PL": ["train"],  # translation not trained on
        "NQ-NL": ["train"],  # translation not trained on
        "NFCorpus": ["train"],
        "FEVERHardNegatives": ["train"],
        "NanoFEVERRetrieval": ["train"],
        "FEVER-NL": ["train"],  # translation not trained on
        "FiQA2018-NL": ["train"],  # translation not trained on
        "BiorxivClusteringP2P.v2": ["train"],
        "BiorxivClusteringS2S.v2": ["train"],
        "MedrxivClusteringP2P.v2": ["train"],
        "MedrxivClusteringS2S.v2": ["train"],
        "MSMARCO": ["train"],
        "MSMARCOHardNegatives": ["train"],
        "NanoMSMARCORetrieval": ["train"],
        "MSMARCO-PL": ["train"],  # translation not trained on
        "mMARCO-NL": ["train"],  # translation not trained on
        "HotpotQA-PL": ["train"],  # translation not trained on
        "HotpotQA-NL": ["train"],  # translation not trained on
        "HotpotQAHardNegatives": ["train"],
        "FEVER": ["train"],
        "FiQA2018": ["train"],
        "DBPedia": ["train"],
        "TRECCOVID": ["train"],
        "ArxivClusteringP2P.v2": ["train"],
        "STSBenchmarkMultilingualSTS": ["train"],  # translated, not trained on
    },
    public_training_code=None,
    public_training_data=None,
)
