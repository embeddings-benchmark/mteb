from __future__ import annotations

import time
from functools import partial, wraps
from typing import Any, Literal

import numpy as np

from mteb.model_meta import ModelMeta
from mteb.models.text_formatting_utils import corpus_to_texts
from mteb.requires_package import requires_package


def rate_limit(max_rpm: int):
    interval = 60 / max_rpm

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_time = time.time()

            result = func(*args, **kwargs)

            elapsed_time = current_time - time.time()
            if elapsed_time < interval:
                time.sleep(interval - elapsed_time)
            return result

        return wrapper

    return decorator


class VoyageWrapper:
    def __init__(
        self,
        model_name: str,
        max_retries: int = 5,
        max_rpm: int = 3,
        **kwargs,
    ) -> None:
        requires_package(self, "voyageai", "Voyage")
        import voyageai

        self._client = voyageai.Client(max_retries=max_retries)
        self._model_name = model_name
        self._max_rpm = max_rpm

    def encode(
        self, sentences: list[str], *, batch_size: int = 32, **kwargs: Any
    ) -> np.ndarray:
        return self._batched_encode(sentences, batch_size, "document")

    def encode_queries(
        self, queries: list[str], *, batch_size: int = 32, **kwargs: Any
    ) -> np.ndarray:
        return self._batched_encode(queries, batch_size, "query")

    def encode_corpus(
        self,
        corpus: list[dict[str, str]] | dict[str, list[str]],
        *,
        batch_size: int = 32,
        **kwargs: Any,
    ) -> np.ndarray:
        sentences = corpus_to_texts(corpus)
        return self._batched_encode(sentences, batch_size, "document")

    def _batched_encode(
        self,
        sentences: list[str],
        batch_size: int,
        input_type: Literal["query", "document"],
    ) -> np.ndarray:
        embeddings = []

        for i in range(0, len(sentences), batch_size):
            embeddings.extend(
                rate_limit(self._max_rpm)(self._client.embed)(
                    sentences[i : i + batch_size],
                    model=self._model_name,
                    input_type=input_type,
                ).embeddings
            )
        return np.array(embeddings)


voyage_large_2_instruct = ModelMeta(
    name="voyage-large-2-instruct",
    revision="1",
    release_date="2024-05-05",
    languages=None,  # supported languages not specified
    loader=partial(VoyageWrapper, model_name="voyage-large-2-instruct"),
    max_tokens=16000,
    embed_dim=1024,
    open_source=False,
)

voyage_finance_2 = ModelMeta(
    name="voyage-finance-2",
    revision="1",
    release_date="2024-05-30",
    languages=None,  # supported languages not specified
    loader=partial(VoyageWrapper, model_name="voyage-finance-2"),
    max_tokens=32000,
    embed_dim=1024,
    open_source=False,
)

voyage_law_2 = ModelMeta(
    name="voyage-law-2",
    revision="1",
    release_date="2024-04-15",
    languages=None,  # supported languages not specified
    loader=partial(VoyageWrapper, model_name="voyage-law-2"),
    max_tokens=16000,
    embed_dim=1024,
    open_source=False,
)

voyage_code_2 = ModelMeta(
    name="voyage-code-2",
    revision="1",
    release_date="2024-01-23",
    languages=None,  # supported languages not specified
    loader=partial(VoyageWrapper, model_name="voyage-code-2"),
    max_tokens=16000,
    embed_dim=1536,
    open_source=False,
)

voyage_large_2 = ModelMeta(
    name="voyage-large-2",  # The release date is considered to be the date of publication of this post https://blog.voyageai.com/2023/10/29/voyage-embeddings/
    revision="1",
    release_date="2023-10-29",
    languages=None,  # supported languages not specified
    loader=partial(VoyageWrapper, model_name="voyage-large-2"),
    max_tokens=16000,
    embed_dim=1536,
    open_source=False,
)

voyage_2 = ModelMeta(
    name="voyage-2",
    revision="1",
    release_date="2023-10-29",
    languages=None,  # supported languages not specified
    loader=partial(VoyageWrapper, model_name="voyage-2"),
    max_tokens=4000,
    embed_dim=1024,
    open_source=False,
)
