import logging
from collections.abc import Callable
from typing import Any

import torch
from torch.utils.data import DataLoader

from mteb.abstasks.task_metadata import TaskMetadata
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.models.models_protocols import EncoderProtocol
from mteb.types import Array, BatchedInput, PromptType

from .repllama_models import RepLLaMAModel, model_prompts

logger = logging.getLogger(__name__)


class PromptrieverModel(RepLLaMAModel, AbsEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def encode(
        self,
        inputs: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> Array:
        kwargs["is_promptriever"] = True
        return super().encode(
            inputs,
            task_metadata=task_metadata,
            hf_split=hf_split,
            hf_subset=hf_subset,
            prompt_type=prompt_type,
            **kwargs,
        )


def _loader(
    wrapper: type[PromptrieverModel], **kwargs
) -> Callable[..., EncoderProtocol]:
    _kwargs = kwargs

    def loader_inner(**kwargs: Any) -> EncoderProtocol:
        return wrapper(**_kwargs, **kwargs)

    return loader_inner


PROMPTRIEVER_CITATION = """
@article{weller2024promptriever,
      title={Promptriever: Instruction-Trained Retrievers Can Be Prompted Like Language Models},
      author={Orion Weller and Benjamin Van Durme and Dawn Lawrie and Ashwin Paranjape and Yuhao Zhang and Jack Hessel},
      year={2024},
      eprint={2409.11136},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2409.11136},
}
"""


promptriever_llama2 = ModelMeta(
    loader=_loader(
        PromptrieverModel,
        base_model_name_or_path="meta-llama/Llama-2-7b-hf",
        peft_model_name_or_path="samaya-ai/promptriever-llama2-7b-v1",
        device_map="auto",
        torch_dtype=torch.bfloat16,
        model_prompts=model_prompts,
    ),
    name="samaya-ai/promptriever-llama2-7b-v1",
    model_type=["dense"],
    languages=["eng-Latn"],
    open_weights=True,
    revision="01c7f73d771dfac7d292323805ebc428287df4f9-30b14e3813c0fa45facfd01a594580c3fe5ecf23",  # base-peft revision
    release_date="2024-09-15",
    n_parameters=7_000_000_000,
    memory_usage_mb=26703,
    max_tokens=4096,
    embed_dim=4096,
    license="apache-2.0",
    training_datasets=set(
        # "samaya-ai/msmarco-w-instructions"
    ),
    reference="https://huggingface.co/samaya-ai/promptriever-llama2-7b-v1",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["PyTorch", "Tevatron"],
    use_instructions=True,
    citation=PROMPTRIEVER_CITATION,
    public_training_code=None,
    public_training_data=None,
)

promptriever_llama3 = ModelMeta(
    loader=_loader(
        PromptrieverModel,
        base_model_name_or_path="meta-llama/Meta-Llama-3.1-8B",
        peft_model_name_or_path="samaya-ai/promptriever-llama3.1-8b-v1",
        device_map="auto",
        torch_dtype=torch.bfloat16,
        model_prompts=model_prompts,
    ),
    name="samaya-ai/promptriever-llama3.1-8b-v1",
    model_type=["dense"],
    languages=["eng-Latn"],
    open_weights=True,
    revision="48d6d0fc4e02fb1269b36940650a1b7233035cbb-2ead22cfb1b0e0c519c371c63c2ab90ffc511b8a",  # base-peft revision
    training_datasets={
        # "samaya-ai/msmarco-w-instructions",
        "mMARCO-NL",  # translation not trained on
    },
    release_date="2024-09-15",
    n_parameters=8_000_000_000,
    memory_usage_mb=30518,
    max_tokens=8192,
    embed_dim=4096,
    license="apache-2.0",
    reference="https://huggingface.co/samaya-ai/promptriever-llama3.1-8b-v1",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["PyTorch", "Tevatron"],
    use_instructions=True,
    citation=PROMPTRIEVER_CITATION,
    public_training_code=None,
    public_training_data=None,
)

promptriever_llama3_instruct = ModelMeta(
    loader=_loader(
        PromptrieverModel,
        base_model_name_or_path="meta-llama/Meta-Llama-3.1-8B-Instruct",
        peft_model_name_or_path="samaya-ai/promptriever-llama3.1-8b-instruct-v1",
        device_map="auto",
        torch_dtype=torch.bfloat16,
        model_prompts=model_prompts,
    ),
    name="samaya-ai/promptriever-llama3.1-8b-instruct-v1",
    model_type=["dense"],
    languages=["eng-Latn"],
    open_weights=True,
    revision="5206a32e0bd3067aef1ce90f5528ade7d866253f-8b677258615625122c2eb7329292b8c402612c21",  # base-peft revision
    release_date="2024-09-15",
    n_parameters=8_000_000_000,
    memory_usage_mb=30518,
    max_tokens=8192,
    embed_dim=4096,
    training_datasets={
        # "samaya-ai/msmarco-w-instructions",
        "mMARCO-NL",  # translation not trained on
    },
    license="apache-2.0",
    reference="https://huggingface.co/samaya-ai/promptriever-llama3.1-8b-instruct-v1",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["PyTorch", "Tevatron"],
    use_instructions=True,
    citation=PROMPTRIEVER_CITATION,
    public_training_code=None,
    public_training_data=None,
)

promptriever_mistral_v1 = ModelMeta(
    loader=_loader(
        PromptrieverModel,
        base_model_name_or_path="mistralai/Mistral-7B-v0.1",
        peft_model_name_or_path="samaya-ai/promptriever-mistral-v0.1-7b-v1",
        device_map="auto",
        torch_dtype=torch.bfloat16,
        model_prompts=model_prompts,
    ),
    name="samaya-ai/promptriever-mistral-v0.1-7b-v1",
    model_type=["dense"],
    languages=["eng-Latn"],
    open_weights=True,
    revision="7231864981174d9bee8c7687c24c8344414eae6b-876d63e49b6115ecb6839893a56298fadee7e8f5",  # base-peft revision
    release_date="2024-09-15",
    n_parameters=7_000_000_000,
    memory_usage_mb=26703,
    training_datasets={
        # "samaya-ai/msmarco-w-instructions",
        "mMARCO-NL",  # translation not trained on
    },
    max_tokens=4096,
    embed_dim=4096,
    license="apache-2.0",
    reference="https://huggingface.co/samaya-ai/promptriever-mistral-v0.1-7b-v1",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["PyTorch", "Tevatron"],
    use_instructions=True,
    citation=PROMPTRIEVER_CITATION,
    public_training_code=None,
    public_training_data=None,
)
