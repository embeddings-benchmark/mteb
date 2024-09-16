from __future__ import annotations

import logging
from typing import Any, Callable, Literal

import numpy as np
import torch

from mteb.encoder_interface import Encoder
from mteb.model_meta import ModelMeta

from .repllama_models import RepLLaMAWrapper

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

EncodeTypes = Literal["query", "passage"]


class PromptrieverWrapper(RepLLaMAWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def encode_queries(self, queries: list[str], **kwargs: Any) -> np.ndarray:
        queries = [f"query:  {query}" for query in queries]
        if "instruction" in kwargs:
            end_punct_list = [
                "?" if query.strip()[-1] not in ["?", ".", "!"] else ""
                for query in queries
            ]
            queries = [
                f"{query}{end_punct_list[i]} {kwargs['instruction']}"
                for i, query in enumerate(queries)
            ]
        return self.encode(queries, **kwargs)


def _loader(wrapper: type[PromptrieverWrapper], **kwargs) -> Callable[..., Encoder]:
    _kwargs = kwargs

    def loader_inner(**kwargs: Any) -> Encoder:
        return wrapper(**_kwargs, **kwargs)

    return loader_inner


promptriever_llama2 = ModelMeta(
    loader=_loader(
        RepLLaMAWrapper,
        base_model_name_or_path="meta-llama/Llama-2-7b-hf",
        peft_model_name_or_path="samaya-ai/promptriever-llama2-7b-v1",
        device_map="auto",
        torch_dtype=torch.bfloat16,
    ),
    name="samaya-ai/promptriever-llama2-7b-v1",
    languages=["eng_Latn"],
    open_source=True,
    revision=None,  # TODO: Not sure what to put here as a model is made of two peft repos, each with a different revision
    release_date="2024-09-15",
)

promptriever_llama3 = ModelMeta(
    loader=_loader(
        RepLLaMAWrapper,
        base_model_name_or_path="meta-llama/Meta-Llama-3.1-8B",
        peft_model_name_or_path="samaya-ai/promptriever-llama3.1-8b-v1",
        device_map="auto",
        torch_dtype=torch.bfloat16,
    ),
    name="samaya-ai/promptriever-llama3.1-8b-v1",
    languages=["eng_Latn"],
    open_source=True,
    revision=None,  # TODO: Not sure what to put here as a model is made of two peft repos, each with a different revision
    release_date="2024-09-15",
)


promptriever_llama3_instruct = ModelMeta(
    loader=_loader(
        RepLLaMAWrapper,
        base_model_name_or_path="meta-llama/Meta-Llama-3.1-8B-Instruct",
        peft_model_name_or_path="samaya-ai/promptriever-llama3.1-8b-instruct-v1",
        device_map="auto",
        torch_dtype=torch.bfloat16,
    ),
    name="samaya-ai/promptriever-llama3.1-8b-instruct-v1",
    languages=["eng_Latn"],
    open_source=True,
    revision=None,  # TODO: Not sure what to put here as a model is made of two peft repos, each with a different revision
    release_date="2024-09-15",
)

promptriever_mistral_v1 = ModelMeta(
    loader=_loader(
        RepLLaMAWrapper,
        base_model_name_or_path="mistralai/Mistral-7B-v0.1",
        peft_model_name_or_path="samaya-ai/promptriever-mistral-v0.1-7b-v1",
        device_map="auto",
        torch_dtype=torch.bfloat16,
    ),
    name="samaya-ai/promptriever-mistral-v0.1-7b-v1",
    languages=["eng_Latn"],
    open_source=True,
    revision=None,  # TODO: Not sure what to put here as a model is made of two peft repos, each with a different revision
    release_date="2024-09-15",
)

promptriever_mistral_v3 = ModelMeta(
    loader=_loader(
        RepLLaMAWrapper,
        base_model_name_or_path="mistralai/Mistral-7B-v0.3",
        peft_model_name_or_path="samaya-ai/promptriever-mistral-v0.3-7b-v1",
        device_map="auto",
        torch_dtype=torch.bfloat16,
    ),
    name="samaya-ai/promptriever-mistral-v0.3-7b-v1",
    languages=["eng_Latn"],
    open_source=True,
    revision=None,  # TODO: Not sure what to put here as a model is made of two peft repos, each with a different revision
    release_date="2024-09-15",
)
