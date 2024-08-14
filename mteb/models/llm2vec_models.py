from __future__ import annotations

import logging
from typing import Any, Callable, Literal

import numpy as np
import torch

from mteb.encoder_interface import Encoder
from mteb.model_meta import ModelMeta
from mteb.models.text_formatting_utils import corpus_to_texts

from .instructions import task_to_instruction

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

EncodeTypes = Literal["query", "passage"]


def llm2vec_instruction(instruction):
    if len(instruction) > 0 and instruction[-1] != ":":
        instruction = instruction.strip(".") + ":"
    return instruction


class LLM2VecWrapper:
    def __init__(self, *args, **kwargs):
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
        self.task_to_instructions = None
        if "task_to_instructions" in kwargs:
            self.task_to_instructions = kwargs.pop("task_to_instructions")

        if "device" in kwargs:
            kwargs["device_map"] = kwargs.pop("device")
        elif torch.cuda.device_count() > 1:
            # bug fix for multi-gpu
            kwargs["device_map"] = None

        self.model = LLM2Vec.from_pretrained(*args, **extra_kwargs, **kwargs)

    def encode(
        self,
        sentences: list[str],
        *,
        prompt_name: str = None,
        **kwargs: Any,  # noqa
    ) -> np.ndarray:
        if prompt_name is not None:
            instruction = (
                self.task_to_instructions[prompt_name]
                if self.task_to_instructions
                and prompt_name in self.task_to_instructions
                else llm2vec_instruction(task_to_instruction(prompt_name))
            )
        else:
            instruction = ""

        sentences = [[instruction, sentence] for sentence in sentences]
        return self.model.encode(sentences, **kwargs)

    def encode_corpus(
        self,
        corpus: list[dict[str, str]] | dict[str, list[str]] | list[str],
        prompt_name: str = None,
        **kwargs: Any,
    ) -> np.ndarray:
        sentences = corpus_to_texts(corpus, sep=" ")
        sentences = [["", sentence] for sentence in sentences]
        if "request_qid" in kwargs:
            kwargs.pop("request_qid")
        return self.model.encode(sentences, **kwargs)

    def encode_queries(self, queries: list[str], **kwargs: Any) -> np.ndarray:
        return self.encode(queries, **kwargs)


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
