from __future__ import annotations

from functools import partial
from typing import Any

import torch
from sentence_transformers import SentenceTransformer

from mteb.model_meta import ModelMeta
from mteb.models.text_formatting_utils import corpus_to_texts

from .instructions import task_to_instruction


class SFRWrapper:
    """Follow the implementation from https://huggingface.co/Salesforce/SFR-Embedding-2_R"""

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
        if "prompt_name" in kwargs:
            instruction = task_to_instruction(
                kwargs.pop("prompt_name"), kwargs.get("is_query", True)
            )
            sentences = [self.get_detailed_instruct(instruction, q) for q in sentences]
        return self.mdl.encode(sentences, batch_size=batch_size, **kwargs)

    def get_detailed_instruct(self, task_description: str, query: str) -> str:
        return f"Instruct: {task_description}\nQuery: {query}"

    def encode_queries(self, queries: list[str], batch_size: int = 32, **kwargs: Any):
        instruction = ""
        if "instruction" in kwargs:
            instruction = kwargs.pop("instruction")

        sentences = [self.get_detailed_instruct(instruction, q) for q in queries]
        emb = self.encode(sentences, batch_size=batch_size, **kwargs)
        return emb

    def encode_corpus(
        self,
        corpus: list[dict[str, str]] | dict[str, list[str]],
        batch_size: int = 32,
        **kwargs: Any,
    ):
        kwargs["is_query"] = False
        sentences = corpus_to_texts(corpus)
        emb = self.encode(sentences, batch_size=batch_size, **kwargs)
        return emb


SFR_Embedding_2_R = ModelMeta(
    loader=partial(SFRWrapper, model_name="Salesforce/SFR-Embedding-2_R"),
    name="Salesforce/SFR-Embedding-2_R",
    languages=["eng_Latn"],
    open_source=True,
    revision="33888956c27c1f0a14edc7f8412c54ca54bb54c3",
    release_date="2024-06-14",  # initial commit of hf model.
)


if __name__ == "__main__":
    import mteb

    mdl = mteb.get_model(SFR_Embedding_2_R.name, SFR_Embedding_2_R.revision)
    emb = mdl.encode(["Hello, world!"])
