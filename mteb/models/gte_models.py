from __future__ import annotations

from functools import partial
from typing import Any

import torch

from mteb.model_meta import ModelMeta
from mteb.models.text_formatting_utils import corpus_to_texts

try:
    from .instructions import task_to_instruction
except:
    from instructions import task_to_instruction


def gte_loader(**kwargs):
    try:
        from gritlm import GritLM
        class GTEWrapper(GritLM):
            def get_detailed_instruct(self, instruction: str, query: str) -> str:
                return f"Instruct: {instruction}\nQuery: "

            def encode(self, *args, **kwargs):
                instruction = ""
                if ("prompt_name" in kwargs) and (kwargs.get("is_query", True)):
                    instruction = self.get_detailed_instruct(
                        task_to_instruction(kwargs.pop("prompt_name"))
                    )
                kwargs["instruction"] = instruction
                return super().encode(*args, **kwargs)

            def encode_corpus(self, *args, **kwargs):
                kwargs["is_query"] = False
                return super().encode_corpus(*args, **kwargs)
    except ImportError:
        raise ImportError(
            "Please install `pip install gritlm` to use gte-Qwen2-7B-instruct."
        )
    kwargs.pop("device", None)  # GritLM does automatic device placement
    return GTEWrapper(**kwargs)

gte_Qwen2_7B_instruct = ModelMeta(
    loader=partial(
        gte_loader,
        model_name_or_path="Alibaba-NLP/gte-Qwen2-7B-instruct",
        attn="cccc",
        pooling_method="lasttoken",
        mode="embedding",
        torch_dtype="auto",
        # The ST script does not normalize while the HF one does so unclear what to do
        # https://huggingface.co/Alibaba-NLP/gte-Qwen2-7B-instruct#sentence-transformers
        normalized=True,
    ),
    name="Alibaba-NLP/gte-Qwen2-7B-instruct",
    languages=None,
    open_source=True,
    revision="e26182b2122f4435e8b3ebecbf363990f409b45b",
    release_date="2024-06-15",  # initial commit of hf model.
)


if __name__ == "__main__":
    import mteb

    mdl = mteb.get_model(gte_Qwen2_7B_instruct.name, gte_Qwen2_7B_instruct.revision)
    emb = mdl.encode(["Hello, world!"])
