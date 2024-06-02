from __future__ import annotations

from typing import Any

import torch
from sentence_transformers import SentenceTransformer

from mteb.model_meta import ModelMeta


class E5Wrapper:
    """following the implementation within the Scandinavian Embedding Benchmark and the intfloat/multilingual-e5-small documentation."""

    def __init__(self, model_name: str, sep: str = " "):
        self.model_name = model_name
        self.mdl = SentenceTransformer(model_name)
        self.sep = sep

    def to(self, device: torch.device) -> None:
        self.mdl.to(device)

    def encode(  # type: ignore
        self,
        sentences: list[str],
        *,
        batch_size: int = 32,
        **kwargs: Any,
    ):
        return self.encode_queries(sentences, batch_size=batch_size, **kwargs)

    def encode_queries(self, queries: list[str], batch_size: int = 32, **kwargs: Any):
        sentences = ["query: " + sentence for sentence in queries]
        emb = self.mdl.encode(sentences, batch_size=batch_size, **kwargs)
        return emb

    def encode_corpus(
        self, corpus: list[dict[str, str]], batch_size: int = 32, **kwargs: Any
    ):
        if isinstance(corpus, dict):
            sentences = [
                (corpus["title"][i] + self.sep + corpus["text"][i]).strip()  # type: ignore
                if "title" in corpus
                else corpus["text"][i].strip()  # type: ignore
                for i in range(len(corpus["text"]))  # type: ignore
            ]
        else:
            sentences = [
                (doc["title"] + self.sep + doc["text"]).strip()
                if "title" in doc
                else doc["text"].strip()
                for doc in corpus
            ]
        sentences = ["passage: " + sentence for sentence in sentences]
        emb = self.mdl.encode(sentences, batch_size=batch_size, **kwargs)
        return emb


e5_mult_small = ModelMeta(
    loader=lambda: E5Wrapper("intfloat/multilingual-e5-small"),  # type: ignore
    name="intfloat/multilingual-e5-small",
    languages=[],  # TODO: missing
    open_source=True,
    revision="e4ce9877abf3edfe10b0d82785e83bdcb973e22e",  # TODO: update revision
    release_date="2021-08-30",  # TODO: add release date
)
