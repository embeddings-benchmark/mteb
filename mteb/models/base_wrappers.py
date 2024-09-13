from __future__ import annotations

from typing import Callable

import torch
from sentence_transformers import SentenceTransformer

from mteb.models.instructions import task_to_instruction
from mteb.models.text_formatting_utils import corpus_to_texts


class SentenceTransformerWrapper:
    def __init__(
        self,
        model_name: str,
        query_instruction: Callable[[str], str] | None,
        corpus_instruction: Callable[[str], str] | None,
    ) -> None:
        self.model_name = model_name
        self.mdl = SentenceTransformer(model_name)
        self._query_instruction = query_instruction
        self._corpus_instruction = corpus_instruction

    def to(self, device: torch.device) -> None:
        self.mdl.to(device)

    def encode(self, sentences: list[str], **kwargs):
        if "prompt_name" in kwargs:
            if "instruction" in kwargs:
                raise ValueError("Cannot specify both `prompt_name` and `instruction`.")
            instruction = task_to_instruction(
                kwargs.pop("prompt_name"), kwargs.pop("is_query", True)
            )
        else:
            instruction = kwargs.pop("instruction", "")
        if instruction:
            if kwargs.pop("is_query", True):
                instruction_func = self._query_instruction
            else:
                instruction_func = self._corpus_instruction
            if instruction_func:
                sentences = [instruction_func(instruction) + s for s in sentences]

        kwargs.pop("prompt_name", None)
        kwargs.pop("request_qid", None)
        kwargs.pop("is_query", None)

        return self.mdl.encode(sentences, **kwargs)

    def encode_queries(self, queries: list[str], **kwargs):
        kwargs["is_query"] = True
        return self.encode(sentences=queries, **kwargs)

    def encode_corpus(self, corpus: list[dict[str, str]] | dict[str, list[str]], **kwargs):
        kwargs["is_query"] = False
        sentences = corpus_to_texts(corpus)
        return self.encode(sentences=sentences, **kwargs)
