from __future__ import annotations

import time
from functools import partial, wraps
from typing import Any, Literal

import numpy as np

from mteb.encoder_interface import PromptType
from mteb.model_meta import ModelMeta
from mteb.models.sentence_transformer_wrapper import get_prompt_name
from mteb.requires_package import requires_package


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


class VoyageWrapper:
    def __init__(
        self,
        model_name: str,
        max_retries: int = 5,
        max_rpm: int = 300,
        max_tpm: int = 1_000_000,
        task_to_prompt: dict[str, str] | None = None,
        **kwargs,
    ) -> None:
        requires_package(self, "voyageai", "Voyage")
        import voyageai

        self._client = voyageai.Client(max_retries=max_retries)
        self._embed_func = rate_limit(max_rpm)(token_limit(max_tpm)(self._client.embed))
        self._model_name = model_name
        self._max_tpm = max_tpm
        self.task_to_prompt = task_to_prompt

    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int = 32,
        task_name: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        input_type = (
            get_prompt_name(self.task_to_prompt, task_name, prompt_type) or "document"
        )
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


prompt_params = {
    "task_to_prompt": {
        PromptType.query.value: "query",
        PromptType.passage.value: "document",
    }
}

voyage_large_2_instruct = ModelMeta(
    name="voyage-large-2-instruct",
    revision="1",
    release_date="2024-05-05",
    languages=None,  # supported languages not specified
    loader=partial(
        VoyageWrapper, model_name="voyage-large-2-instruct", **prompt_params
    ),
    max_tokens=16000,
    embed_dim=1024,
    open_source=False,
)

voyage_finance_2 = ModelMeta(
    name="voyage-finance-2",
    revision="1",
    release_date="2024-05-30",
    languages=None,  # supported languages not specified
    loader=partial(VoyageWrapper, model_name="voyage-finance-2", **prompt_params),
    max_tokens=32000,
    embed_dim=1024,
    open_source=False,
)

voyage_law_2 = ModelMeta(
    name="voyage-law-2",
    revision="1",
    release_date="2024-04-15",
    languages=None,  # supported languages not specified
    loader=partial(VoyageWrapper, model_name="voyage-law-2", **prompt_params),
    max_tokens=16000,
    embed_dim=1024,
    open_source=False,
)

voyage_code_2 = ModelMeta(
    name="voyage-code-2",
    revision="1",
    release_date="2024-01-23",
    languages=None,  # supported languages not specified
    loader=partial(VoyageWrapper, model_name="voyage-code-2", **prompt_params),
    max_tokens=16000,
    embed_dim=1536,
    open_source=False,
)

voyage_large_2 = ModelMeta(
    name="voyage-large-2",  # Date of publication of this post https://blog.voyageai.com/2023/10/29/voyage-embeddings/
    revision="1",
    release_date="2023-10-29",
    languages=None,  # supported languages not specified
    loader=partial(VoyageWrapper, model_name="voyage-large-2", **prompt_params),
    max_tokens=16000,
    embed_dim=1536,
    open_source=False,
)

voyage_2 = ModelMeta(
    name="voyage-2",
    revision="1",
    release_date="2023-10-29",
    languages=None,  # supported languages not specified
    loader=partial(VoyageWrapper, model_name="voyage-2", **prompt_params),
    max_tokens=4000,
    embed_dim=1024,
    open_source=False,
)
# see https://blog.voyageai.com/2024/06/10/voyage-multilingual-2-multilingual-embedding-model/"
voyage_multilingual_2 = ModelMeta(
    name="voyage-multilingual-2",
    revision="1",
    release_date="2024-06-10",
    languages=None,  # supported languages not specified
    loader=partial(VoyageWrapper, model_name="voyage-multilingual-2", **prompt_params),
    max_tokens=32000,
    embed_dim=1024,
    open_source=False,
)
