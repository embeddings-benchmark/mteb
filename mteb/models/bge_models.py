from __future__ import annotations

from functools import partial
from typing import Any

import torch
from sentence_transformers import SentenceTransformer

from mteb.model_meta import ModelMeta
from mteb.models.text_formatting_utils import corpus_to_texts


class BGEWrapper:
    """following the hf model card documentation."""

    def __init__(self, model_name: str, **kwargs: Any):
        self.model_name = model_name
        self.mdl = SentenceTransformer(model_name)

    def to(self, device: torch.device) -> None:
        self.mdl.to(device)

    def encode(  # type: ignore
        self,
        sentences: list[str],
        *,
        batch_size: int = 32,
        **kwargs: Any,
    ):
        return self.mdl.encode(sentences, batch_size=batch_size, **kwargs)

    def encode_queries(self, queries: list[str], batch_size: int = 32, **kwargs: Any):
        if "prompt_name" in kwargs:
            kwargs.pop("prompt_name")
        sentences = [
            "Represent this sentence for searching relevant passages: " + sentence
            for sentence in queries
        ]
        emb = self.mdl.encode(
            sentences, batch_size=batch_size, normalize_embeddings=True, **kwargs
        )
        return emb

    def encode_corpus(
        self,
        corpus: list[dict[str, str]] | dict[str, list[str]],
        batch_size: int = 32,
        **kwargs: Any,
    ):
        if "prompt_name" in kwargs:
            kwargs.pop("prompt_name")
        sentences = corpus_to_texts(corpus)
        emb = self.mdl.encode(
            sentences, batch_size=batch_size, normalize_embeddings=True, **kwargs
        )
        return emb


bge_base_en_v1_5 = ModelMeta(
    loader=partial(BGEWrapper, model_name="BAAI/bge-base-en-v1.5"),  # type: ignore
    name="BAAI/bge-base-en-v1.5",
    languages=["eng_Latn"],
    open_source=True,
    revision="a5beb1e3e68b9ab74eb54cfd186867f64f240e1a",
    release_date="2023-09-11",  # initial commit of hf model.
)

bge_large_en_v1_5 = ModelMeta(
    loader=partial(BGEWrapper, model_name="BAAI/bge-large-en-v1.5"),  # type: ignore
    name="BAAI/bge-large-en-v1.5",
    languages=["eng_Latn"],
    open_source=True,
    revision="d4aa6901d3a41ba39fb536a557fa166f842b0e09",
    release_date="2023-09-12",  # initial commit of hf model.
)
