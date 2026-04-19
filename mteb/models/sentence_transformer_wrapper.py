from __future__ import annotations

import logging
import sys
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from packaging.version import Version

from mteb._log_once import LogOnce
from mteb.models import ModelMeta
from mteb.types import OutputDType, PromptType

from .abs_encoder import AbsEncoder

if sys.version_info >= (3, 13):
    from warnings import deprecated
else:
    from typing_extensions import deprecated

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder, SentenceTransformer
    from torch.utils.data import DataLoader
    from typing_extensions import Unpack

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput, EncodeKwargs

logger = logging.getLogger(__name__)

SENTENCE_TRANSFORMERS_QUERY_ENCODE_VERSION = "5.0.0"


@deprecated(
    "sentence_transformers_loader is deprecated, use SentenceTransformerEncoderWrapper directly instead."
)
def sentence_transformers_loader(
    model_name: str, revision: str | None = None, device: str | None = None, **kwargs
) -> SentenceTransformerEncoderWrapper:
    """Loads a SentenceTransformer model and wraps it in a SentenceTransformerEncoderWrapper.

    .. deprecated:: 2.11.0
        Use :class:`SentenceTransformerEncoderWrapper` directly instead.

    Args:
        model_name: The name of the SentenceTransformer model to load.
        revision: The revision of the model to load.
        device: The device used to load the model.
        kwargs: Additional arguments to pass to the SentenceTransformer model.
    """
    return SentenceTransformerEncoderWrapper(
        model=model_name, revision=revision, device=device, **kwargs
    )


class SentenceTransformerEncoderWrapper(AbsEncoder):
    """Multimodal wrapper for SentenceTransformer models."""

    mteb_model_meta: ModelMeta

    def __init__(
        self,
        model: str | SentenceTransformer,
        revision: str | None = None,
        device: str | None = None,
        model_prompts: dict[str, str] | None = None,
        *,
        embed_dim: int | None = None,
        **kwargs,
    ) -> None:
        """Wrapper for SentenceTransformer models.

        Args:
            model: The SentenceTransformer model to use. Can be a string (model name), a SentenceTransformer model, or a CrossEncoder model.
            revision: The revision of the model to use.
            device: The device used to load the model.
            model_prompts: A dictionary mapping task names to prompt names.
                First priority is given to the composed prompt of task name + prompt type (query or passage), then to the specific task prompt,
                then to the composed prompt of task type + prompt type, then to the specific task type prompt,
                and finally to the specific prompt type.
            embed_dim: The embedding dimension of the model to use.
            **kwargs: Additional arguments to pass to the SentenceTransformer model.
        """
        from sentence_transformers import SentenceTransformer

        if isinstance(model, str):
            self.model = SentenceTransformer(
                model,
                revision=revision,
                device=device,
                truncate_dim=embed_dim,
                **kwargs,
            )
            self.mteb_model_meta = ModelMeta.create_empty(
                overwrites=dict(
                    name=model,
                    revision=revision,
                    loader=sentence_transformers_loader,
                )
            )
        else:
            self.model = model
            self.mteb_model_meta = ModelMeta.from_sentence_transformer_model(self.model)

        built_in_prompts = getattr(self.model, "prompts", None)
        if built_in_prompts and not model_prompts:
            model_prompts = built_in_prompts
        elif model_prompts and built_in_prompts:
            msg = f"Model prompts specified, these will overwrite the default model prompts. Current prompts will be:\n {model_prompts}"
            logger.warning(msg)
            warnings.warn(msg)
            self.model.prompts = model_prompts

        self.model_prompts, invalid_prompts = self.validate_task_to_prompt_name(
            model_prompts, raise_for_invalid_keys=False
        )

        if invalid_prompts:
            invalid_prompts = "\n".join(invalid_prompts)
            msg = f"Some prompts are not in the expected format and will be ignored. Problems:\n\n{invalid_prompts}"
            logger.warning(msg)
            warnings.warn(msg)

        if (
            self.model_prompts
            and len(self.model_prompts) <= 2
            and (
                PromptType.query.value not in self.model_prompts
                or PromptType.document.value not in self.model_prompts
            )
        ):
            msg = f"SentenceTransformers that use prompts most often need to be configured with at least 'query' and 'document' prompts to ensure optimal performance. Received {self.model_prompts}"
            logger.warning(msg)
            warnings.warn(msg)

    def similarity(self, embeddings1: Array, embeddings2: Array) -> Array:
        """Compute the similarity between two collections of embeddings."""
        if hasattr(self.model, "similarity") and callable(self.model.similarity):
            return self.model.similarity(embeddings1, embeddings2)
        return super().similarity(embeddings1, embeddings2)

    def encode(
        self,
        inputs: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        **kwargs: Unpack[EncodeKwargs],
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
        from sentence_transformers import __version__ as st_version

        if "precision" in kwargs:
            existing_experiment_kwargs = self.mteb_model_meta.experiment_kwargs
            output_dtype = OutputDType.from_str(kwargs["precision"])  # type: ignore[typeddict-item]
            if existing_experiment_kwargs is not None:
                existing_experiment_kwargs["output_dtypes"] = output_dtype  # type: ignore[index]
            else:
                existing_experiment_kwargs = {"output_dtypes": output_dtype.value}
            logger.warning(
                f"The 'precision' argument passed in encode_kwargs setting output_dtypes to {output_dtype.value}."
            )
            self.mteb_model_meta = self.mteb_model_meta.model_copy(
                update={
                    "experiment_kwargs": existing_experiment_kwargs,
                },
                deep=True,
            )

        has_query_encode = (
            Version(st_version).release
            >= Version(SENTENCE_TRANSFORMERS_QUERY_ENCODE_VERSION).release
        )

        _inputs = [text for batch in inputs for text in batch["text"]]

        prompt = None
        prompt_name = None
        if self.model_prompts is not None:
            prompt_name = self.get_prompt_name(task_metadata, prompt_type)
            prompt = self.model_prompts.get(prompt_name, None)  # type: ignore[arg-type]
        if prompt_name:
            prompt_log = f"Using {prompt_name=} for task={task_metadata.name} {prompt_type=} with {prompt=}"
        else:
            prompt_log = (
                f"No model prompts found for task={task_metadata.name} {prompt_type=}"
            )

        LogOnce(logger).info(prompt_log)
        logger.debug(f"Encoding {len(_inputs)} sentences.")

        if prompt_type and has_query_encode:
            if prompt_type == PromptType.query:
                encode_function = self.model.encode_query
            elif prompt_type == PromptType.document:
                encode_function = self.model.encode_document
            else:
                raise ValueError(f"Unknown prompt type: {prompt_type}")
        else:
            encode_function = self.model.encode

        embeddings = encode_function(
            _inputs,
            prompt=prompt,
            **kwargs,
        )
        if isinstance(embeddings, torch.Tensor):
            # ensure everything is on CPU and is float
            embeddings = embeddings.cpu().detach().float()
        return embeddings


class SentenceTransformerMultimodalEncoderWrapper(SentenceTransformerEncoderWrapper):
    """Wrapper for multimodal SentenceTransformer models."""

    def __init__(
        self,
        *args,
        fps: float | None = None,
        max_frames: int | None = None,
        num_frames: int | None = None,
        target_sampling_rate: int | None = None,
        max_samples: int | None = None,
        **kwargs,
    ) -> None:
        """Wrapper for multimodal SentenceTransformer models.

        Args:
            *args: Passed to SentenceTransformerEncoderWrapper.
            fps: Target frames per second for video sampling.
            max_frames: Safety cap on frames per video for FPS mode.
            num_frames: If set, use fixed-sample mode instead of FPS-based.
            target_sampling_rate: Sampling rate to resample audio to.
            max_samples: Maximum number of audio samples to keep.
            **kwargs: Passed to SentenceTransformerEncoderWrapper.
        """
        super().__init__(*args, **kwargs)
        self.fps = fps
        self.max_frames = max_frames
        self.num_frames = num_frames
        self.target_sampling_rate = target_sampling_rate
        self.max_samples = max_samples

    def encode(
        self,
        inputs: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        **kwargs: Unpack[EncodeKwargs],
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
        has_video = "video" in inputs.dataset.features
        has_audio = "audio" in inputs.dataset.features
        if has_video:
            from mteb._create_dataloaders import VideoCollator

            inputs.collate_fn = VideoCollator(
                target_sampling_rate=self.target_sampling_rate or 16000,
                fps=self.fps,
                max_frames=self.max_frames,
                num_frames=self.num_frames,
                max_samples=self.max_samples,
            )
        elif has_audio:
            from mteb._create_dataloaders import AudioCollator

            inputs.collate_fn = AudioCollator(
                target_sampling_rate=self.target_sampling_rate or 16000,
                max_samples=self.max_samples,
            )

        prompt = None
        prompt_name = None
        if self.model_prompts is not None:
            prompt_name = self.get_prompt_name(task_metadata, prompt_type)
            prompt = self.model_prompts.get(prompt_name, None)  # type: ignore[arg-type]
        if prompt_name:
            logger.info(
                f"Using {prompt_name=} for task={task_metadata.name} {prompt_type=} with {prompt=}"
            )
        else:
            logger.info(
                f"No model prompts found for task={task_metadata.name} {prompt_type=}"
            )

        all_embeddings = []
        _modality_keys = {"text", "image", "audio", "video"}
        for batch in inputs:
            # Transformers' apply_chat_template expects audio as raw numpy arrays,
            # not the {"array", "sampling_rate"} dict produced by AudioCollator.
            # See https://github.com/huggingface/sentence-transformers/issues/3732
            if "audio" in batch:
                batch["audio"] = [
                    a["array"] if isinstance(a, dict) and "array" in a else a
                    for a in batch["audio"]
                ]
            batch_column = next(iter(batch.keys()))
            batched_input: list[dict[str, Any]] = [
                dict() for _ in range(len(batch[batch_column]))
            ]

            # transform from {"text": [text1, text2], "image": [image1, image2]} to
            # [{"text": text1, "image": image1}, {"text": text2, "image": image2}]
            # Only pass through recognized modality keys; ST rejects unknown keys.
            for key, values in batch.items():
                if key not in _modality_keys:
                    continue
                for i, value in enumerate(values):
                    batched_input[i][key] = value

            embeddings = self.model.encode(
                batched_input,
                prompt=prompt,
                **kwargs,
            )
            if isinstance(embeddings, torch.Tensor):
                # ensure everything is on CPU and is float
                embeddings = embeddings.cpu().detach().float()
            all_embeddings.append(embeddings)
        return np.concatenate(all_embeddings, axis=0)


class CrossEncoderWrapper:
    """Wrapper for CrossEncoder models.

    Args:
        model: The CrossEncoder model to use. Can be a string (model name) or a CrossEncoder model.
        revision: The revision of the model to use.
        device: The device used to load the model.
        query_prefix: A prefix to add to all queries.
        passage_prefix: A prefix to add to all passages.
        **kwargs: Additional arguments to pass to the CrossEncoder model.
    """

    def __init__(
        self,
        model: CrossEncoder | str,
        revision: str | None = None,
        device: str | None = None,
        query_prefix: str = "",
        passage_prefix: str = "",
        **kwargs,
    ) -> None:
        from sentence_transformers import CrossEncoder

        if isinstance(model, CrossEncoder):
            self.model = model
            self.mteb_model_meta = ModelMeta.from_cross_encoder(self.model)
        elif isinstance(model, str):
            self.model = CrossEncoder(model, revision=revision, device=device, **kwargs)
            self.mteb_model_meta = ModelMeta.create_empty(
                overwrites=dict(
                    name=model,
                    revision=revision,
                    loader=CrossEncoderWrapper,
                )
            )
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix

    def predict(
        self,
        inputs1: DataLoader[BatchedInput],
        inputs2: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        **kwargs: Unpack[EncodeKwargs],
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
        all_queries_with_instructions = [
            self.query_prefix + text for batch in inputs1 for text in batch["text"]
        ]
        all_corpus_with_instructions = [
            self.passage_prefix + text for batch in inputs2 for text in batch["text"]
        ]

        return self.model.predict(
            list(zip(all_queries_with_instructions, all_corpus_with_instructions)),
            **kwargs,
        )
