import logging
from typing import Any, Literal

import torch
from torch.utils.data import DataLoader
from vllm.config import PoolerConfig

from mteb.abstasks.task_metadata import TaskMetadata
from mteb.models.abs_encoder import AbsEncoder
from mteb.types import Array, BatchedInput, PromptType

logger = logging.getLogger(__name__)

Dtype = Literal["half", "float16", "float", "float32", "bfloat16"]


def vllm_loader(
    model_name: str, revision: str | None = None, **kwargs
) -> "VllmEncoderWrapper":
    """Loads a model and wraps it in a VllmEncoderWrapper.

    Args:
        model_name: The name of the  to load.
        revision: The revision of the model to load.
        kwargs: Additional arguments to pass to the SentenceTransformer model.
    """
    return VllmEncoderWrapper(model=model_name, revision=revision, **kwargs)


class VllmEncoderWrapper(AbsEncoder):
    """Wrapper for vllm serving engine."""

    def __init__(
        self,
        model: str,
        revision: str | None = None,
        trust_remote_code: bool = True,
        dtype: Dtype = "auto",
        head_dtype: Literal["model"] | Dtype | None = None,
        max_model_len: int | None = None,
        max_num_batched_tokens: int = None,
        max_num_seqs: int = 128,
        tensor_parallel_size: int = 1,
        data_parallel_size: int = 1,
        enable_prefix_caching: bool | None = None,
        gpu_memory_utilization: float = 0.9,
        hf_overrides: dict[str, Any] | None = None,
        pooler_config: PoolerConfig | None = None,
        enforce_eager: bool = False,
        model_prompts: dict[str, str] | None = None,
        **kwargs,
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
            data_parallel_size: Number of replicas for data parallelism. For small models,
                using data_parallel is better than tensor_parallel.
            enable_prefix_caching: Whether to enable KV cache sharing for common prompt prefixes.
                If None, uses the model's default setting.
            gpu_memory_utilization: Target GPU memory utilization ratio (0.0 to 1.0).
            hf_overrides: Dictionary mapping Hugging Face configuration keys to override values.
            pooler_config: Controls the behavior of output pooling in pooling models..
            enforce_eager: Whether to disable CUDA graph optimization and use eager execution.
            model_prompts: A dictionary mapping task names to prompt names.
                First priority is given to the composed prompt of task name + prompt type (query or passage), then to the specific task prompt,
                then to the composed prompt of task type + prompt type, then to the specific task type prompt,
                and finally to the specific prompt type.
            **kwargs: Additional arguments to pass to the vllm serving engine model.
        """
        from vllm import LLM, EngineArgs

        hf_overrides = {} if hf_overrides is None else hf_overrides

        if head_dtype is not None:
            hf_overrides["head_dtype"] = head_dtype

        args = EngineArgs(
            model=model,
            revision=revision,
            runner="pooling",
            convert="embed",
            max_model_len=max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=max_num_seqs,
            tensor_parallel_size=tensor_parallel_size,
            data_parallel_size=data_parallel_size,
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
        self.model_prompts = model_prompts

        if (
            self.model_prompts
            and len(self.model_prompts) <= 2
            and (
                PromptType.query.value not in self.model_prompts
                or PromptType.document.value not in self.model_prompts
            )
        ):
            logger.warning(
                "Encode models use prompts most often need to be configured with at least 'query' and"
                f" 'document' prompts to ensure optimal performance. Received {self.model_prompts}"
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

            The order of priorities for prompt selection are:
                1. Composed prompt of task name + prompt type (query or passage)
                2. Specific task prompt
                3. Composed prompt of task type + prompt type (query or passage)
                4. Specific task type prompt
                5. Specific prompt type (query or passage)


        Returns:
            The encoded sentences.
        """

        prompt = None
        prompt_name = None
        if self.model_prompts is not None:
            prompt_name = self.get_prompt_name(task_metadata, prompt_type)
            prompt = self.model_prompts.get(prompt_name, None)
        if prompt_name:
            logger.info(
                f"Using {prompt_name=} for task={task_metadata.name} {prompt_type=} with {prompt=}"
            )
        else:
            logger.info(
                f"No model prompts found for task={task_metadata.name} {prompt_type=}"
            )

        # TODO: support task prompt
        prompts = [text for batch in inputs for text in batch["text"]]
        outputs = self.llm.encode(
            prompts, pooling_task="embed", truncate_prompt_tokens=-1
        )
        embeddings = torch.stack([output.outputs.data for output in outputs])
        return embeddings

    def cleanup(self):
        """Clean up the VLLM distributed runtime environment and release GPU resources."""
        if self.llm is None:
            return

        import gc

        from vllm.distributed import cleanup_dist_env_and_memory

        self.llm = None
        gc.collect()
        cleanup_dist_env_and_memory()

    def __del__(self):
        self.cleanup()
