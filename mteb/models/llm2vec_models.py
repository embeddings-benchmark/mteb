from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np
import torch

from mteb.encoder_interface import Encoder, PromptType
from mteb.model_meta import ModelMeta

from .instructions import task_to_instruction
from .sentence_transformer_wrapper import validate_task_to_prompt_name
from .wrapper import Wrapper

logger = logging.getLogger(__name__)


def llm2vec_instruction(instruction):
    if len(instruction) > 0 and instruction[-1] != ":":
        instruction = instruction.strip(".") + ":"
    return instruction


class LLM2VecWrapper(Wrapper):
    def __init__(
        self,
        model_prompts: dict[str, str] | None = None,
        device: str | None = None,
        *args,
        **kwargs,
    ):
        try:
            from llm2vec import LLM2Vec
        except ImportError:
            raise ImportError(
                "To use the LLM2Vec models `llm2vec` is required. Please install it with `pip install llm2vec`."
            )
        extra_kwargs = {}
        try:
            import flash_attn  # noqa

            extra_kwargs["attn_implementation"] = "flash_attention_2"
        except ImportError:
            logger.warning(
                "LLM2Vec models were trained with flash attention enabled. For optimal performance, please install the `flash_attn` package with `pip install flash-attn --no-build-isolation`."
            )
        self.model_prompts = (
            validate_task_to_prompt_name(model_prompts) if model_prompts else None
        )

        if device:
            kwargs["device_map"] = device
        elif torch.cuda.device_count() > 1:
            # bug fix for multi-gpu
            kwargs["device_map"] = None

        self.model = LLM2Vec.from_pretrained(*args, **extra_kwargs, **kwargs)

    def encode(
        self,
        sentences: list[str],
        *,
        task_name: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,  # noqa
    ) -> np.ndarray:
        instruction = llm2vec_instruction(
            task_to_instruction(task_name, prompt_type == PromptType.query)
        )

        sentences = [[instruction, sentence] for sentence in sentences]
        return self.model.encode(sentences, **kwargs)


def _loader(wrapper: type[LLM2VecWrapper], **kwargs) -> Callable[..., Encoder]:
    _kwargs = kwargs

    def loader_inner(**kwargs: Any) -> Encoder:
        return wrapper(**_kwargs, **kwargs)

    return loader_inner


llm2vec_llama3_8b_supervised = ModelMeta(
    loader=_loader(
        LLM2VecWrapper,
        base_model_name_or_path="McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
        peft_model_name_or_path="McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised",
        device_map="auto",
        torch_dtype=torch.bfloat16,
    ),
    name="McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised",
    languages=["eng_Latn"],
    open_source=True,
    revision=None,  # TODO: Not sure what to put here as a model is made of two peft repos, each with a different revision
    release_date="2024-04-09",
)

llm2vec_llama3_8b_unsupervised = ModelMeta(
    loader=_loader(
        LLM2VecWrapper,
        base_model_name_or_path="McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
        peft_model_name_or_path="McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-unsup-simcse",
        device_map="auto",
        torch_dtype=torch.bfloat16,
    ),
    name="McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-unsup-simcse",
    languages=["eng_Latn"],
    open_source=True,
    revision=None,
    release_date="2024-04-09",
)


llm2vec_mistral7b_supervised = ModelMeta(
    loader=_loader(
        LLM2VecWrapper,
        base_model_name_or_path="McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp",
        peft_model_name_or_path="McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp-supervised",
        device_map="auto",
        torch_dtype=torch.bfloat16,
    ),
    name="McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp-supervised",
    languages=["eng_Latn"],
    open_source=True,
    revision=None,
    release_date="2024-04-09",
)

llm2vec_mistral7b_unsupervised = ModelMeta(
    loader=_loader(
        LLM2VecWrapper,
        base_model_name_or_path="McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp",
        peft_model_name_or_path="McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp-unsup-simcse",
        device_map="auto",
        torch_dtype=torch.bfloat16,
    ),
    name="McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp-unsup-simcse",
    languages=["eng_Latn"],
    open_source=True,
    revision=None,
    release_date="2024-04-09",
)

llm2vec_llama2_7b_supervised = ModelMeta(
    loader=_loader(
        LLM2VecWrapper,
        base_model_name_or_path="McGill-NLP/LLM2Vec-Llama-2-7b-chat-hf-mntp",
        peft_model_name_or_path="McGill-NLP/LLM2Vec-Llama-2-7b-chat-hf-mntp-supervised",
        device_map="auto",
        torch_dtype=torch.bfloat16,
    ),
    name="McGill-NLP/LLM2Vec-Llama-2-7b-chat-hf-mntp-supervised",
    languages=["eng_Latn"],
    open_source=True,
    revision=None,
    release_date="2024-04-09",
)

llm2vec_llama2_7b_unsupervised = ModelMeta(
    loader=_loader(
        LLM2VecWrapper,
        base_model_name_or_path="McGill-NLP/LLM2Vec-Llama-2-7b-chat-hf-mntp",
        peft_model_name_or_path="McGill-NLP/LLM2Vec-Llama-2-7b-chat-hf-mntp-unsup-simcse",
        device_map="auto",
        torch_dtype=torch.bfloat16,
    ),
    name="McGill-NLP/LLM2Vec-Llama-2-7b-chat-hf-mntp-unsup-simcse",
    languages=["eng_Latn"],
    open_source=True,
    revision=None,
    release_date="2024-04-09",
)

llm2vec_sheared_llama_supervised = ModelMeta(
    loader=_loader(
        LLM2VecWrapper,
        base_model_name_or_path="McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp",
        peft_model_name_or_path="McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp-supervised",
        device_map="auto",
        torch_dtype=torch.bfloat16,
    ),
    name="McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp-supervised",
    languages=["eng_Latn"],
    open_source=True,
    revision=None,
    release_date="2024-04-09",
)

llm2vec_sheared_llama_unsupervised = ModelMeta(
    loader=_loader(
        LLM2VecWrapper,
        base_model_name_or_path="McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp",
        peft_model_name_or_path="McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp-unsup-simcse",
        device_map="auto",
        torch_dtype=torch.bfloat16,
    ),
    name="McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp-unsup-simcse",
    languages=["eng_Latn"],
    open_source=True,
    revision=None,
    release_date="2024-04-09",
)
