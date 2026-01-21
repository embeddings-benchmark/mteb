from __future__ import annotations

import atexit
import gc
import logging
import os
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import torch

from mteb._requires_package import requires_package
from mteb.models import ModelMeta
from mteb.models.abs_encoder import AbsEncoder
from mteb.types import PromptType

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch.utils.data import DataLoader
    from vllm.config import PoolerConfig  # type: ignore[import-not-found]

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput


logger = logging.getLogger(__name__)

Dtype = Literal["half", "float16", "float", "float32", "bfloat16", "auto"]


class VllmWrapperBase:
    """Wrapper for vllm serving engine."""

    convert = "auto"
    mteb_model_meta: ModelMeta | None = None

    def __init__(
        self,
        model: str | ModelMeta,
        revision: str | None = None,
        *,
        trust_remote_code: bool = True,
        dtype: Dtype = "auto",
        head_dtype: Literal["model"] | Dtype | None = None,
        max_model_len: int | None = None,
        max_num_batched_tokens: int | None = None,
        max_num_seqs: int = 128,
        tensor_parallel_size: int = 1,
        enable_prefix_caching: bool | None = None,
        gpu_memory_utilization: float = 0.9,
        hf_overrides: dict[str, Any] | None = None,
        pooler_config: PoolerConfig | None = None,
        enforce_eager: bool = False,
        **kwargs: Any,
    ):
        """Wrapper for vllm serving engine.

        Args:
            model: model name string.
            revision: The revision of the model to use.
            trust_remote_code: Whether to trust remote code execution when loading the model.
                Should be True for models with custom code.
            dtype: Data type for model weights. "auto" will automatically select appropriate
                dtype based on hardware and model capabilities. vllm uses flash attention by
                default, which does not support fp32. Therefore, it defaults to using fp16 for
                inference on fp32 models. Testing has shown a relatively small drop in accuracy.
                You can manually opt for fp32, but inference speed will be very slow.
            head_dtype: "head" refers to the last Linear layer(s) of an LLMs, such as the score
                or classifier in a classification model. Uses fp32 for the head by default to
                gain extra precision.
            max_model_len: Maximum sequence length (context window) supported by the model.
                If None, uses the model's default maximum length.
            max_num_batched_tokens: Maximum number of tokens to process in a single batch.
                If None, automatically determined.
            max_num_seqs: Maximum number of sequences to process concurrently.
            tensor_parallel_size: Number of GPUs for tensor parallelism.
            enable_prefix_caching: Whether to enable KV cache sharing for common prompt prefixes.
                If None, uses the model's default setting.
            gpu_memory_utilization: Target GPU memory utilization ratio (0.0 to 1.0).
            hf_overrides: Dictionary mapping Hugging Face configuration keys to override values.
            pooler_config: Controls the behavior of output pooling in pooling models.
            enforce_eager: Whether to disable CUDA graph optimization and use eager execution.
            **kwargs: Additional arguments to pass to the vllm serving engine model.
        """
        requires_package(
            self,
            "vllm",
            "Wrapper for vllm serving engine",
            install_instruction="pip install mteb[vllm]",
        )

        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

        from vllm import LLM, EngineArgs

        hf_overrides = {} if hf_overrides is None else hf_overrides

        if head_dtype is not None:
            hf_overrides["head_dtype"] = head_dtype

        model_name = model if isinstance(model, str) else model.name

        if isinstance(model, ModelMeta):
            logger.info(
                "Using revision from model meta. Passed revision will be ignored"
            )
            revision = model.revision

        args = EngineArgs(
            model=model_name,
            revision=revision,
            runner="pooling",
            convert=self.convert,  # type: ignore[arg-type]
            max_model_len=max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=max_num_seqs,
            tensor_parallel_size=tensor_parallel_size,
            enable_prefix_caching=enable_prefix_caching,
            gpu_memory_utilization=gpu_memory_utilization,
            hf_overrides=hf_overrides,
            pooler_config=pooler_config,
            enforce_eager=enforce_eager,
            trust_remote_code=trust_remote_code,
            dtype=dtype,
            **kwargs,
        )
        self.llm = LLM(**vars(args))

        if isinstance(model, str):
            self.mteb_model_meta = ModelMeta.from_hub(model=model, revision=revision)
        else:
            self.mteb_model_meta = model

        atexit.register(self.cleanup)

    def cleanup(self):
        """Clean up the VLLM distributed runtime environment and release GPU resources."""
        if self.llm is None:
            return

        from vllm.distributed import (  # type: ignore[import-not-found]
            cleanup_dist_env_and_memory,
        )

        self.llm = None
        gc.collect()
        cleanup_dist_env_and_memory()

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass


class VllmEncoderWrapper(AbsEncoder, VllmWrapperBase):
    """vLLM wrapper for Encoder models.

    Args:
        model: model name string or ModelMeta.
        revision: The revision of the model to use.
        prompt_dict: A dictionary mapping task names to prompt strings.
        use_instructions: Whether to use instructions from the prompt_dict.
            When False, values from prompt_dict are used as static prompts (prefixes).
            When True, values from prompt_dict are used as instructions to be formatted
            using the instruction_template.
        instruction_template: A template or callable to format instructions.
            Can be a string with '{instruction}' placeholder or a callable that takes
            the instruction and prompt type and returns a formatted string.
        apply_instruction_to_documents: Whether to apply instructions to documents prompts.
        **kwargs: Additional arguments to pass to the vllm serving engine model.
    """

    convert = "embed"

    def __init__(
        self,
        model: str | ModelMeta,
        revision: str | None = None,
        prompt_dict: dict[str, str] | None = None,
        use_instructions: bool = False,
        instruction_template: (
            str | Callable[[str, PromptType | None], str] | None
        ) = None,
        apply_instruction_to_documents: bool = True,
        **kwargs: Any,
    ):
        if use_instructions and instruction_template is None:
            raise ValueError(
                "To use instructions, an instruction_template must be provided. "
                "For example, `Instruction: {instruction}`"
            )

        if (
            isinstance(instruction_template, str)
            and "{instruction}" not in instruction_template
        ):
            raise ValueError(
                "Instruction template must contain the string '{instruction}'."
            )

        self.prompts_dict = prompt_dict
        self.use_instructions = use_instructions
        self.instruction_template = instruction_template
        self.apply_instruction_to_passages = apply_instruction_to_documents
        super().__init__(
            model,
            revision,
            **kwargs,
        )

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
        """Encodes the given sentences using the encoder.

        Args:
            inputs: The sentences to encode.
            task_metadata: The metadata of the task. Sentence-transformers uses this to
                determine which prompt to use from a specified dictionary.
            prompt_type: The name type of prompt. (query or passage)
            hf_split: Split of current task
            hf_subset: Subset of current task
            **kwargs: Additional arguments to pass to the encoder.

        Returns:
            The encoded sentences.
        """
        prompt = ""
        if self.use_instructions and self.prompts_dict is not None:
            prompt = self.get_task_instruction(task_metadata, prompt_type)
        elif self.prompts_dict is not None:
            prompt_name = self.get_prompt_name(task_metadata, prompt_type)
            if prompt_name is not None:
                prompt = self.prompts_dict.get(prompt_name, "")

        if (
            self.use_instructions
            and self.apply_instruction_to_passages is False
            and prompt_type == PromptType.document
        ):
            logger.info(
                f"No instruction used, because prompt type = {prompt_type.document}"
            )
            prompt = ""
        else:
            logger.info(
                f"Using instruction: '{prompt}' for task: '{task_metadata.name}' prompt type: '{prompt_type}'"
            )

        prompts = [prompt + text for batch in inputs for text in batch["text"]]
        outputs = self.llm.encode(
            prompts, pooling_task="embed", truncate_prompt_tokens=-1
        )
        embeddings = torch.stack([output.outputs.data for output in outputs])
        return embeddings


class VllmCrossEncoderWrapper(VllmWrapperBase):
    """vLLM wrapper for CrossEncoder models."""

    convert = "classify"

    def __init__(
        self,
        model: str | ModelMeta,
        revision: str | None = None,
        query_prefix: str = "",
        document_prefix: str = "",
        **kwargs: Any,
    ):
        super().__init__(
            model,
            revision,
            **kwargs,
        )
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix

    def predict(
        self,
        inputs1: DataLoader[BatchedInput],
        inputs2: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> Array:
        """Predicts relevance scores for pairs of inputs. Note that, unlike the encoder, the cross-encoder can compare across inputs.

        Args:
            inputs1: First Dataloader of inputs to encode. For reranking tasks, these are queries (for text only tasks `QueryDatasetType`).
            inputs2: Second Dataloader of inputs to encode. For reranking, these are documents (for text only tasks `RetrievalOutputType`).
            task_metadata: Metadata of the current task.
            hf_split: Split of current task, allows to know some additional information about current split.
                E.g. Current language
            hf_subset: Subset of current task. Similar to `hf_split` to get more information
            prompt_type: The name type of prompt. (query or passage)
            **kwargs: Additional arguments to pass to the cross-encoder.

        Returns:
            The predicted relevance scores for each inputs pair.
        """
        queries = [
            self.query_prefix + text for batch in inputs1 for text in batch["text"]
        ]
        corpus = [
            self.document_prefix + text for batch in inputs2 for text in batch["text"]
        ]
        # TODO: support score prompt

        outputs = self.llm.score(
            queries,
            corpus,
            truncate_prompt_tokens=-1,
            use_tqdm=False,
        )
        scores = np.array([output.outputs.score for output in outputs])
        return scores
