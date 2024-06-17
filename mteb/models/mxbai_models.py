from __future__ import annotations

from functools import partial
from typing import Any

import torch
from sentence_transformers import SentenceTransformer

from mteb.model_meta import ModelMeta
from mteb.models.text_formatting_utils import corpus_to_texts


class MxbaiWrapper:
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
        sentences = [
            "Represent this sentence for searching relevant passages: " + sentence
            for sentence in queries
        ]
        emb = self.mdl.encode(sentences, batch_size=batch_size, **kwargs)
        return emb

    def encode_corpus(
        self,
        corpus: list[dict[str, str]] | dict[str, list[str]],
        batch_size: int = 32,
        **kwargs: Any,
    ):
        sentences = corpus_to_texts(corpus)
        emb = self.mdl.encode(sentences, batch_size=batch_size, **kwargs)
        return emb


mxbai_embed_large_v1 = ModelMeta(
    loader=partial(MxbaiWrapper, model_name="mixedbread-ai/mxbai-embed-large-v1"),  # type: ignore
    name="mixedbread-ai/mxbai-embed-large-v1",
    languages=["eng_Latn"],
    open_source=True,
    revision="990580e27d329c7408b3741ecff85876e128e203",
    release_date="2024-03-07",  # initial commit of hf model.
)
