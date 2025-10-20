import logging
from collections.abc import Callable
from typing import Any

import torch
from torch.utils.data import DataLoader

from mteb._requires_package import requires_package
from mteb.abstasks.task_metadata import TaskMetadata
from mteb.types import Array, BatchedInput, PromptType

from .abs_encoder import AbsEncoder

logger = logging.getLogger(__name__)


def instruct_wrapper(
    model_name_or_path: str,
    mode: str,
    instruction_template: str | Callable[[str], str] | None = None,
    **kwargs,
):
    """Instruct wrapper for models. Uses GritLM to pass instructions to the model.

    It's recommended to use `InstructSentenceTransformerModel` instead of this wrapper for models.

    Args:
        model_name_or_path: Model name or path.
        mode: Mode of the model. Either 'query' or 'passage'.
        instruction_template: Instruction template. Should contain the string '{instruction}'.
        **kwargs: Additional arguments to pass to the model.
    """
    requires_package(
        instruct_wrapper, "gritlm", model_name_or_path, "pip install 'mteb[gritlm]'"
    )
    from gritlm import GritLM  # type: ignore[import]

    class InstructGritLMModel(GritLM, AbsEncoder):
        def __init__(
            self,
            model_name_or_path: str,
            mode: str,
            instruction_template: str | Callable[[str, PromptType], str] | None = None,
            **kwargs,
        ):
            if (
                isinstance(instruction_template, str)
                and "{instruction}" not in instruction_template
            ):
                raise ValueError(
                    "Instruction template must contain the string '{instruction}'."
                )
            if instruction_template is None:
                logger.warning(
                    "No instruction template provided. Instructions will be used as-is."
                )

            if "gte-Qwen" in model_name_or_path:
                logger.warning(
                    "Instructions are used in both query and docs, which may cause performance discrepancies from the original implementation."
                )

            self.instruction_template = instruction_template
            super().__init__(model_name_or_path=model_name_or_path, mode=mode, **kwargs)

        def encode(
            self,
            inputs: DataLoader[BatchedInput],
            *args,
            task_metadata: TaskMetadata,
            hf_split: str,
            hf_subset: str,
            prompt_type: PromptType | None = None,
            **kwargs: Any,
        ) -> Array:
            instruction = self.get_instruction(task_metadata, prompt_type)

            if self.instruction_template:
                instruction = self.format_instruction(instruction, prompt_type)
            _inputs = [text for batch in inputs for text in batch["text"]]

            logger.info(
                f"Using instruction: '{instruction}' for task: '{task_metadata.name}'"
            )
            embeddings = super().encode(
                _inputs, instruction=instruction, *args, **kwargs
            )
            if isinstance(embeddings, torch.Tensor):
                # sometimes in kwargs can be return_tensors=True
                embeddings = embeddings.cpu().detach().float().numpy()
            return embeddings

    return InstructGritLMModel(model_name_or_path, mode, instruction_template, **kwargs)


class InstructSentenceTransformerModel(AbsEncoder):
    """Instruction wrapper for Sentence Transformer models."""

    def __init__(
        self,
        model_name: str,
        revision: str,
        instruction_template: str
        | Callable[[str, PromptType | None], str]
        | None = None,
        max_seq_length: int | None = None,
        apply_instruction_to_passages: bool = True,
        padding_side: str | None = None,
        add_eos_token: bool = False,
        prompts_dict: dict[str, str] | None = None,
        **kwargs: Any,
    ):
        """Instruct Sentence Transformer Wrapper. Wrapper that passes instructions to the Sentence Transformer model.

        Applied for models like e5-instruct, jasper, etc.

        Arguments:
            model_name: Model name of the sentence transformers model.
            revision: Revision of the sentence transformers model.
            instruction_template: Model template. Should contain the string '{instruction}'.
            max_seq_length: Maximum sequence length. If None, the maximum sequence length will be read from the model config.
            apply_instruction_to_passages: Whether to apply the instruction template to the passages.
            padding_side: Padding side. If None, the padding side will be read from the model config.
            add_eos_token: Whether to add the eos token to each input example.
            prompts_dict: Dictionary of task names to prompt names. If None, the prompts will be read from the model config.
            **kwargs: Kwargs for Sentence Transformer model.
        """
        from sentence_transformers import SentenceTransformer

        if (
            isinstance(instruction_template, str)
            and "{instruction}" not in instruction_template
        ):
            raise ValueError(
                "Instruction template must contain the string '{instruction}'."
            )
        if instruction_template is None:
            logger.warning(
                "No instruction template provided. Instructions will be used as-is."
            )

        self.instruction_template = instruction_template
        tokenizer_params = {}
        if add_eos_token:
            tokenizer_params["add_eos_token"] = add_eos_token
        if max_seq_length is not None:
            # https://github.com/UKPLab/sentence-transformers/blob/7341bf155b4349b88690b78c84beb5aa658c439f/sentence_transformers/models/Transformer.py#L115
            tokenizer_params["model_max_length"] = max_seq_length
        if padding_side is not None:
            tokenizer_params["padding_side"] = padding_side

        kwargs.setdefault("tokenizer_kwargs", {}).update(tokenizer_params)

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, revision=revision, **kwargs)
        self.apply_instruction_to_passages = apply_instruction_to_passages
        self.prompts_dict = prompts_dict

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
            inputs: Batch of inputs to encode.
            task_metadata: The metadata of the task. Encoders (e.g. SentenceTransformers) use to
                select the appropriate prompts, with priority given to more specific task/prompt combinations over general ones.

                The order of priorities for prompt selection are:
                    1. Composed prompt of task name + prompt type (query or passage)
                    2. Specific task prompt
                    3. Composed prompt of task type + prompt type (query or passage)
                    4. Specific task type prompt
                    5. Specific prompt type (query or passage)
            hf_split: Split of current task, allows to know some additional information about current split.
                E.g. Current language
            hf_subset: Subset of current task. Similar to `hf_split` to get more information
            prompt_type: The name type of prompt. (query or passage)
            **kwargs: Additional arguments to pass to the encoder.

        Returns:
            The encoded input in a numpy array or torch tensor of the shape (Number of sentences) x (Embedding dimension).
        """
        sentences = [text for batch in inputs for text in batch["text"]]
        instruction = self.get_task_instruction(task_metadata, prompt_type)

        # to passage prompts won't be applied to passages
        if (
            not self.apply_instruction_to_passages
            and prompt_type == PromptType.document
        ):
            instruction = None
            logger.info(
                f"No instruction used, because prompt type = {prompt_type.document}"
            )

        if instruction:
            logger.info(
                f"Using instruction: '{instruction}' for task: '{task_metadata.name}'"
            )

        embeddings = self.model.encode(
            sentences,
            prompt=instruction,
            **kwargs,
        )

        if isinstance(embeddings, torch.Tensor):
            # sometimes in kwargs can be return_tensors=True
            embeddings = embeddings.cpu().detach().float().numpy()
        return embeddings
