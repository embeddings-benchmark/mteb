from __future__ import annotations

from functools import partial
from typing import Any

import numpy as np
import torch

from mteb.import_utils import requires_package
from mteb.model_meta import ModelMeta
from mteb.models.utils import corpus_to_texts


class VoyageWrapper:
    def __init__(self, model_name: str) -> None:
        requires_package(self, "voyageai", "Voyage")
        import voyageai

        self._client = voyageai.Client()
        self._model_name = model_name

    def encode(self, sentences: list[str], **kwargs: Any) -> torch.Tensor | np.ndarray:
        return np.array(
            self._client.embed(
                sentences, self._model_name, input_type="document"
            ).embeddings
        )

    def encode_queries(
        self, queries: list[str], **kwargs: Any
    ) -> torch.Tensor | np.ndarray:
        return np.array(
            self._client.embed(queries, self._model_name, input_type="query").embeddings
        )

    def encode_corpus(
        self, corpus: list[dict[str, str]] | dict[str, list[str]], **kwargs: Any
    ) -> torch.Tensor | np.ndarray:
        sentences = corpus_to_texts(corpus)
        return np.array(
            self._client.embed(
                sentences, self._model_name, input_type="document"
            ).embeddings
        )


voyage_large_2_instruct = ModelMeta(
    name="voyage-large-2-instruct",
    revision="1",
    release_date="2024-05-05",
    languages=["eng-Latn"],
    loader=partial(VoyageWrapper, model_name="voyage-large-2-instruct"),
    max_tokens=16000,
    embed_dim=1024,
    open_source=False,
)

voyage_finance_2 = ModelMeta(
    name="voyage-finance-2",
    revision="1",
    release_date="2024-05-30",
    languages=["eng-Latn"],
    loader=partial(VoyageWrapper, model_name="voyage-finance-2"),
    max_tokens=32000,
    embed_dim=1024,
    open_source=False,
)

voyage_law_2 = ModelMeta(
    name="voyage-law-2",
    revision="1",
    release_date="2024-04-15",
    languages=["eng-Latn"],
    loader=partial(VoyageWrapper, model_name="voyage-law-2"),
    max_tokens=16000,
    embed_dim=1024,
    open_source=False,
)

voyage_code_2 = ModelMeta(
    name="voyage-code-2",
    revision="1",
    release_date="2024-01-23",
    languages=["eng-Latn"],
    loader=partial(VoyageWrapper, model_name="voyage-code-2"),
    max_tokens=16000,
    embed_dim=1536,
    open_source=False,
)

voyage_large_2 = ModelMeta(
    name="voyage-large-2",
    revision="1",
    release_date=None,
    languages=["eng-Latn"],
    loader=partial(VoyageWrapper, model_name="voyage-large-2"),
    max_tokens=16000,
    embed_dim=1536,
    open_source=False,
)

voyage_2 = ModelMeta(
    name="voyage-2",
    revision="1",
    release_date=None,
    languages=["eng-Latn"],
    loader=partial(VoyageWrapper, model_name="voyage-2"),
    max_tokens=4000,
    embed_dim=1024,
    open_source=False,
)
