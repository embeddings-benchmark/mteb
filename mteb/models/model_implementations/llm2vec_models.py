import logging
from collections.abc import Callable
from typing import Any

import torch
from torch.utils.data import DataLoader

from mteb._requires_package import requires_package, suggest_package
from mteb.abstasks.task_metadata import TaskMetadata
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.models.models_protocols import EncoderProtocol
from mteb.types import Array, BatchedInput, PromptType

logger = logging.getLogger(__name__)


def llm2vec_instruction(instruction):
    if len(instruction) > 0 and instruction[-1] != ":":
        instruction = instruction.strip(".") + ":"
    return instruction


llm2vec_supervised_training_data = {
    # source, section g1: https://arxiv.org/pdf/2404.05961
    # splits assumed but unknown
    "HotpotQA",
    "HotpotQA-PL",  # translation not trained on
    "HotpotQA-NL",  # translation not trained on
    "HotpotQAHardNegatives",
    "MSMARCO",
    "MSMARCOHardNegatives",
    "NanoMSMARCORetrieval",
    "MSMARCO-PL",  # translation not trained on
    "mMARCO-NL",  # translation not trained on
    "MIRACLRetrieval",
    "MIRACLRetrievalHardNegatives",
    "MIRACLReranking",
    "NQ",
    "NQHardNegatives",
    "NanoNQRetrieval",
    "NQ-PL",  # translation not trained on
    "NQ-NL",  # translation not trained on
    "FEVER",
    "FEVERHardNegatives",
    "NanoFEVERRetrieval",
    "FEVER-NL",  # translation not trained on
    "MrTidyRetrieval",
    "T2Reranking",
}


class LLM2VecModel(AbsEncoder):
    def __init__(
        self,
        model_prompts: dict[str, str] | None = None,
        device: str | None = None,
        *args,
        **kwargs,
    ):
        model_name = kwargs.get("model_name", "LLM2Vec")
        requires_package(self, "llm2vec", model_name, "pip install 'mteb[llm2vec]'")
        from llm2vec import LLM2Vec

        extra_kwargs = {}
        if suggest_package(
            self,
            "flash_attn",
            model_name,
            "pip install flash-attn --no-build-isolation",
        ):
            import flash_attn  # noqa: F401

            extra_kwargs["attn_implementation"] = "flash_attention_2"

        self.model_prompts = self.validate_task_to_prompt_name(model_prompts)

        if device:
            kwargs["device_map"] = device
        elif torch.cuda.device_count() > 1:
            # bug fix for multi-gpu
            kwargs["device_map"] = None

        self.model = LLM2Vec.from_pretrained(*args, **extra_kwargs, **kwargs)

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
        instruction = llm2vec_instruction(
            self.get_instruction(task_metadata, prompt_type)
        )
        sentences = [text for batch in inputs for text in batch["text"]]

        sentences = [[instruction, sentence] for sentence in sentences]
        return self.model.encode(sentences, **kwargs)


def _loader(wrapper: type[LLM2VecModel], **kwargs) -> Callable[..., EncoderProtocol]:
    _kwargs = kwargs

    def loader_inner(**kwargs: Any) -> EncoderProtocol:
        return wrapper(**_kwargs, **kwargs)

    return loader_inner


LLM2VEC_CITATION = """
@misc{behnamghader2024llm2veclargelanguagemodels,
      title={LLM2Vec: Large Language Models Are Secretly Powerful Text Encoders},
      author={Parishad BehnamGhader and Vaibhav Adlakha and Marius Mosbach and Dzmitry Bahdanau and Nicolas Chapados and Siva Reddy},
      year={2024},
      eprint={2404.05961},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2404.05961},
}"""


llm2vec_llama3_8b_supervised = ModelMeta(
    loader=_loader(
        LLM2VecModel,
        base_model_name_or_path="McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
        peft_model_name_or_path="McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised",
        device_map="auto",
        torch_dtype=torch.bfloat16,
    ),
    name="McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised",
    languages=["eng-Latn"],
    open_weights=True,
    revision="baa8ebf04a1c2500e61288e7dad65e8ae42601a7",
    # TODO: Not sure what to put here as a model is made of two peft repos, each with a different revision
    release_date="2024-04-09",
    n_parameters=7_505_000_000,
    memory_usage_mb=28629,
    max_tokens=8192,
    embed_dim=4096,
    license="mit",
    reference="https://huggingface.co/McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["LLM2Vec", "PyTorch"],
    use_instructions=True,
    public_training_code="https://github.com/McGill-NLP/llm2vec/tree/250292a307428240d801fadd85825464e71c3277/train_configs",
    training_datasets=llm2vec_supervised_training_data,
    public_training_data=None,
    citation=LLM2VEC_CITATION,
)

llm2vec_llama3_8b_unsupervised = ModelMeta(
    loader=_loader(
        LLM2VecModel,
        base_model_name_or_path="McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
        peft_model_name_or_path="McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-unsup-simcse",
        device_map="auto",
        torch_dtype=torch.bfloat16,
    ),
    name="McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-unsup-simcse",
    languages=["eng-Latn"],
    open_weights=True,
    revision="1cb7b735326d13a8541db8f57f35da5373f5e9c6",
    release_date="2024-04-09",
    n_parameters=7_505_000_000,
    memory_usage_mb=28629,
    max_tokens=8192,
    embed_dim=4096,
    license="mit",
    reference="https://huggingface.co/McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-unsup-simcse",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["LLM2Vec", "PyTorch"],
    use_instructions=True,
    citation=LLM2VEC_CITATION,
    public_training_code="https://github.com/McGill-NLP/llm2vec/tree/250292a307428240d801fadd85825464e71c3277/train_configs",
    training_datasets=set(),
    public_training_data=None,
)

llm2vec_mistral7b_supervised = ModelMeta(
    loader=_loader(
        LLM2VecModel,
        base_model_name_or_path="McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp",
        peft_model_name_or_path="McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp-supervised",
        device_map="auto",
        torch_dtype=torch.bfloat16,
    ),
    name="McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp-supervised",
    languages=["eng-Latn"],
    open_weights=True,
    revision="0ae69bdd5816105778b971c3138e8f8a18eaa3ae",
    release_date="2024-04-09",
    n_parameters=7_111_000_000,
    memory_usage_mb=27126,
    max_tokens=32768,
    embed_dim=4096,
    license="mit",
    reference="https://huggingface.co/McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp-supervised",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["LLM2Vec", "PyTorch"],
    use_instructions=True,
    citation=LLM2VEC_CITATION,
    public_training_code="https://github.com/McGill-NLP/llm2vec/tree/250292a307428240d801fadd85825464e71c3277/train_configs",
    training_datasets=llm2vec_supervised_training_data,
    public_training_data=None,
)

llm2vec_mistral7b_unsupervised = ModelMeta(
    loader=_loader(
        LLM2VecModel,
        base_model_name_or_path="McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp",
        peft_model_name_or_path="McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp-unsup-simcse",
        device_map="auto",
        torch_dtype=torch.bfloat16,
    ),
    name="McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp-unsup-simcse",
    languages=["eng-Latn"],
    open_weights=True,
    revision="2c055a5d77126c0d3dc6cd8ffa30e2908f4f45f8",
    release_date="2024-04-09",
    n_parameters=7_111_000_000,
    memory_usage_mb=27126,
    max_tokens=32768,
    embed_dim=4096,
    license="mit",
    reference="https://huggingface.co/McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp-unsup-simcse",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["LLM2Vec", "PyTorch"],
    use_instructions=True,
    citation=LLM2VEC_CITATION,
    public_training_code="https://github.com/McGill-NLP/llm2vec/tree/250292a307428240d801fadd85825464e71c3277/train_configs",
    training_datasets=set(),
    public_training_data=None,
)

llm2vec_llama2_7b_supervised = ModelMeta(
    loader=_loader(
        LLM2VecModel,
        base_model_name_or_path="McGill-NLP/LLM2Vec-Llama-2-7b-chat-hf-mntp",
        peft_model_name_or_path="McGill-NLP/LLM2Vec-Llama-2-7b-chat-hf-mntp-supervised",
        device_map="auto",
        torch_dtype=torch.bfloat16,
    ),
    name="McGill-NLP/LLM2Vec-Llama-2-7b-chat-hf-mntp-supervised",
    languages=["eng-Latn"],
    open_weights=True,
    revision="2c055a5d77126c0d3dc6cd8ffa30e2908f4f45f8",
    release_date="2024-04-09",
    n_parameters=7_111_000_000,
    memory_usage_mb=27126,
    max_tokens=32768,
    embed_dim=4096,
    license="mit",
    reference="https://huggingface.co/McGill-NLP/LLM2Vec-Llama-2-7b-chat-hf-mntp-supervised",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["LLM2Vec", "PyTorch"],
    use_instructions=True,
    citation=LLM2VEC_CITATION,
    public_training_code="https://github.com/McGill-NLP/llm2vec/tree/250292a307428240d801fadd85825464e71c3277/train_configs",
    training_datasets=llm2vec_supervised_training_data,
    public_training_data=None,
)

llm2vec_llama2_7b_unsupervised = ModelMeta(
    loader=_loader(
        LLM2VecModel,
        base_model_name_or_path="McGill-NLP/LLM2Vec-Llama-2-7b-chat-hf-mntp",
        peft_model_name_or_path="McGill-NLP/LLM2Vec-Llama-2-7b-chat-hf-mntp-unsup-simcse",
        device_map="auto",
        torch_dtype=torch.bfloat16,
    ),
    name="McGill-NLP/LLM2Vec-Llama-2-7b-chat-hf-mntp-unsup-simcse",
    languages=["eng-Latn"],
    open_weights=True,
    revision="a76944871d169ebe7c97eb921764cd063afed785",
    release_date="2024-04-09",
    n_parameters=7_111_000_000,
    memory_usage_mb=27126,
    max_tokens=32768,
    embed_dim=4096,
    license="mit",
    reference="https://huggingface.co/McGill-NLP/LLM2Vec-Llama-2-7b-chat-hf-mntp-unsup-simcse",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["LLM2Vec", "PyTorch"],
    use_instructions=True,
    public_training_code="https://github.com/McGill-NLP/llm2vec/tree/250292a307428240d801fadd85825464e71c3277/train_configs",
    training_datasets=set(),
    public_training_data=None,
    citation=LLM2VEC_CITATION,
)

llm2vec_sheared_llama_supervised = ModelMeta(
    loader=_loader(
        LLM2VecModel,
        base_model_name_or_path="McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp",
        peft_model_name_or_path="McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp-supervised",
        device_map="auto",
        torch_dtype=torch.bfloat16,
    ),
    name="McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp-supervised",
    languages=["eng-Latn"],
    open_weights=True,
    revision="a5943d406c6b016fef3f07906aac183cf1a0b47d",
    release_date="2024-04-09",
    n_parameters=7_111_000_000,
    memory_usage_mb=27126,
    max_tokens=32768,
    embed_dim=4096,
    license="mit",
    reference="https://huggingface.co/McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp-supervised",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["LLM2Vec", "PyTorch"],
    use_instructions=True,
    citation=LLM2VEC_CITATION,
    public_training_code="https://github.com/McGill-NLP/llm2vec/tree/250292a307428240d801fadd85825464e71c3277/train_configs",
    training_datasets=llm2vec_supervised_training_data,
    public_training_data=None,
)

llm2vec_sheared_llama_unsupervised = ModelMeta(
    loader=_loader(
        LLM2VecModel,
        base_model_name_or_path="McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp",
        peft_model_name_or_path="McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp-unsup-simcse",
        device_map="auto",
        torch_dtype=torch.bfloat16,
    ),
    name="McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp-unsup-simcse",
    languages=["eng-Latn"],
    open_weights=True,
    revision="a5943d406c6b016fef3f07906aac183cf1a0b47d",
    release_date="2024-04-09",
    n_parameters=7_111_000_000,
    memory_usage_mb=27126,
    max_tokens=32768,
    embed_dim=4096,
    license="mit",
    reference="https://huggingface.co/McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp-unsup-simcse",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["LLM2Vec", "PyTorch"],
    use_instructions=True,
    citation=LLM2VEC_CITATION,
    public_training_code="https://github.com/McGill-NLP/llm2vec/tree/250292a307428240d801fadd85825464e71c3277/train_configs",
    training_datasets=set(),
    public_training_data=None,
)
