from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np
import torch

from mteb.encoder_interface import Encoder
from mteb.model_meta import ModelMeta
from mteb.models.repllama_models import RepLLaMAWrapper
from mteb.models.wrapper import Wrapper

logger = logging.getLogger(__name__)


class PromptrieverWrapper(RepLLaMAWrapper, Wrapper):
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
    languages=["eng-Latn"],
    open_weights=True,
    revision="01c7f73d771dfac7d292323805ebc428287df4f9-30b14e3813c0fa45facfd01a594580c3fe5ecf23",  # base-peft revision
    release_date="2024-09-15",
    n_parameters=7_000_000_000,
    memory_usage_mb=27,
    max_tokens=4096,
    embed_dim=4096,
    license="apache-2.0",
    training_datasets={"samaya-ai/msmarco-w-instructions": ["train"]},
    reference="https://huggingface.co/samaya-ai/promptriever-llama2-7b-v1",
    similarity_fn_name="cosine",
    framework=["PyTorch", "Tevatron"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
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
    languages=["eng-Latn"],
    open_weights=True,
    revision="48d6d0fc4e02fb1269b36940650a1b7233035cbb-2ead22cfb1b0e0c519c371c63c2ab90ffc511b8a",  # base-peft revision
    training_datasets={
        "samaya-ai/msmarco-w-instructions": ["train"],
        "mMARCO-NL": ["train"],  # translation not trained on
    },
    release_date="2024-09-15",
    n_parameters=8_000_000_000,
    memory_usage_mb=31,
    max_tokens=8192,
    embed_dim=4096,
    license="apache-2.0",
    reference="https://huggingface.co/samaya-ai/promptriever-llama3.1-8b-v1",
    similarity_fn_name="cosine",
    framework=["PyTorch", "Tevatron"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
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
    languages=["eng-Latn"],
    open_weights=True,
    revision="5206a32e0bd3067aef1ce90f5528ade7d866253f-8b677258615625122c2eb7329292b8c402612c21",  # base-peft revision
    release_date="2024-09-15",
    n_parameters=8_000_000_000,
    memory_usage_mb=31,
    max_tokens=8192,
    embed_dim=4096,
    training_datasets={
        "samaya-ai/msmarco-w-instructions": ["train"],
        "mMARCO-NL": ["train"],  # translation not trained on
    },
    license="apache-2.0",
    reference="https://huggingface.co/samaya-ai/promptriever-llama3.1-8b-instruct-v1",
    similarity_fn_name="cosine",
    framework=["PyTorch", "Tevatron"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
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
    languages=["eng-Latn"],
    open_weights=True,
    revision="7231864981174d9bee8c7687c24c8344414eae6b-876d63e49b6115ecb6839893a56298fadee7e8f5",  # base-peft revision
    release_date="2024-09-15",
    n_parameters=7_000_000_000,
    memory_usage_mb=27,
    training_datasets={
        "samaya-ai/msmarco-w-instructions": ["train"],
        "mMARCO-NL": ["train"],  # translation not trained on
    },
    max_tokens=4096,
    embed_dim=4096,
    license="apache-2.0",
    reference="https://huggingface.co/samaya-ai/promptriever-mistral-v0.1-7b-v1",
    similarity_fn_name="cosine",
    framework=["PyTorch", "Tevatron"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
)
