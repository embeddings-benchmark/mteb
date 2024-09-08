from __future__ import annotations

from functools import partial
from typing import Any

import torch
from sentence_transformers import SentenceTransformer

from mteb.model_meta import ModelMeta
from mteb.models.text_formatting_utils import corpus_to_texts


class ArcticWrapper:
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


arctic_m_v1_5 = ModelMeta(
    loader=partial(ArcticWrapper, model_name="Snowflake/snowflake-arctic-embed-m-v1.5"),  # type: ignore
    name="Snowflake/snowflake-arctic-embed-m-v1.5",
    languages=["eng_Latn"],
    open_source=True,
    revision="97eab2e17fcb7ccb8bb94d6e547898fa1a6a0f47",
    release_date="2024-07-08",  # initial commit of hf model.
)
