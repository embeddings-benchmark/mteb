from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
from packaging.version import Version
from tqdm.auto import tqdm
from typing_extensions import deprecated

from mteb._log_once import LogOnce
from mteb.models import ModelMeta
from mteb.types import OutputDType, PromptType

from .abs_encoder import AbsEncoder, get_prompt_name

if TYPE_CHECKING:
    from collections.abc import Callable

    from sentence_transformers import CrossEncoder, SentenceTransformer
    from sentence_transformers.sparse_encoder import SparseEncoder
    from torch.utils.data import DataLoader
    from typing_extensions import Unpack

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput, EncodeKwargs, Modalities

logger = logging.getLogger(__name__)

SENTENCE_TRANSFORMERS_QUERY_ENCODE_VERSION = "5.0.0"


@deprecated(
    "sentence_transformers_loader is deprecated, use SentenceTransformerEncoderWrapper directly instead."
)
def sentence_transformers_loader(
    model_name: str,
    revision: str | None = None,
    device: str | None = None,
    **kwargs: Any,
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


def _setup_modality_collator(
    inputs: DataLoader[BatchedInput],
    *,
    fps: float | None,
    max_frames: int | None,
    num_frames: int | None,
    target_sampling_rate: int | None,
    max_samples: int | None,
) -> bool:
    """Attach a VideoCollator/AudioCollator to ``inputs`` if needed.

    Returns True when any modality feature (image/audio/video) is present on
    the dataset so the caller can take the multimodal path.
    """
    features = inputs.dataset.features  # type: ignore[attr-defined]
    has_video = "video" in features
    has_audio = "audio" in features
    if has_video:
        from mteb.models.modality_collators import VideoCollator

        inputs.collate_fn = VideoCollator(
            target_sampling_rate=target_sampling_rate or 16000,
            fps=fps,
            max_frames=max_frames,
            num_frames=num_frames,
            max_samples=max_samples,
        )
    elif has_audio:
        from mteb.models.modality_collators import AudioCollator

        inputs.collate_fn = AudioCollator(
            target_sampling_rate=target_sampling_rate or 16000,
            max_samples=max_samples,
        )
    return has_video or has_audio or "image" in features


def _batch_to_modality_dicts(
    batch: dict[str, Any],
    supported_modalities: list[Modalities],
) -> list[dict[str, Any]]:
    modality_batch = {k: v for k, v in batch.items() if k in supported_modalities}
    LogOnce(logger).info(f"Model will encode modalities {list(modality_batch.keys())}")
    return [
        dict(zip(modality_batch, sample)) for sample in zip(*modality_batch.values())
    ]


def _resolve_model_prompts(
    model: Any, model_prompts: dict[str, str] | None
) -> dict[str, str] | None:
    """Merge `model_prompts` with a sentence-transformers-style `model`'s built-in prompts.

    If only one of the two is given, that one is used as-is. If both are given, `model_prompts`
    takes priority and is written back onto `model.prompts` (with a warning). The merged result is
    then validated, dropping (with a warning) any keys that aren't a valid task name/type or prompt
    type.

    Args:
        model: The sentence-transformers-style model whose built-in `.prompts` (if any) to merge with.
        model_prompts: A dictionary mapping task names to prompt names, as passed to the wrapper.

    Returns:
        The validated, merged prompts dictionary (or None if there are no prompts at all).
    """
    built_in_prompts = getattr(model, "prompts", None)
    if built_in_prompts and not model_prompts:
        model_prompts = built_in_prompts
    elif model_prompts and built_in_prompts:
        msg = f"Model prompts specified, these will overwrite the default model prompts. Current prompts will be:\n {model_prompts}"
        logger.warning(msg)
        warnings.warn(msg)
        model.prompts = model_prompts

    resolved_prompts, invalid_prompts = AbsEncoder.validate_task_to_prompt_name(
        model_prompts, raise_for_invalid_keys=False
    )
    if invalid_prompts:
        invalid_prompts_str = "\n".join(invalid_prompts)
        msg = f"Some prompts are not in the expected format and will be ignored. Problems:\n\n{invalid_prompts_str}"
        logger.warning(msg)
        warnings.warn(msg)
    return resolved_prompts


def _resolve_prompt(
    model_prompts: dict[str, str] | None,
    task_metadata: TaskMetadata,
    prompt_type: PromptType | None,
) -> str | None:
    """Look up the prompt text for `task_metadata`/`prompt_type` in `model_prompts`, logging the outcome.

    Args:
        model_prompts: A dictionary mapping task names to prompt names (see `_resolve_model_prompts`).
        task_metadata: The metadata of the task being encoded.
        prompt_type: The name type of prompt (query or document).

    Returns:
        The prompt text to use, or None if no matching prompt was found.
    """
    prompt = None
    prompt_name = None
    if model_prompts is not None:
        prompt_name = get_prompt_name(model_prompts, task_metadata, prompt_type)
        prompt = model_prompts.get(prompt_name, None)  # type: ignore[arg-type]
    if prompt_name:
        prompt_log = f"Using {prompt_name=} for task={task_metadata.name} {prompt_type=} with {prompt=}"
    else:
        prompt_log = (
            f"No model prompts found for task={task_metadata.name} {prompt_type=}"
        )
    LogOnce(logger).info(prompt_log)
    return prompt


def _select_encode_function(
    model: SentenceTransformer | SparseEncoder,
    prompt_type: PromptType | None,
    *,
    has_query_encode: bool = True,
) -> Callable[..., Any]:
    """Pick `model.encode_query`/`model.encode_document`/`model.encode` based on `prompt_type`.

    Args:
        model: The sentence-transformers-style model to pick the encode method from.
        prompt_type: The name type of prompt (query or document).
        has_query_encode: Whether `model` supports `encode_query`/`encode_document` (added in
            sentence-transformers 5.0). Falls back to `model.encode` when False.

    Returns:
        The bound encode method to use.
    """
    if prompt_type and has_query_encode:
        if prompt_type == PromptType.query:
            return model.encode_query  # type: ignore[no-any-return]
        elif prompt_type == PromptType.document:
            return model.encode_document  # type: ignore[no-any-return]
        raise ValueError(f"Unknown prompt type: {prompt_type}")
    return model.encode


def _postprocess_dense_embeddings(embeddings: Any) -> Any:
    """Move a batch's embeddings to CPU float32 if it's a torch tensor; otherwise pass through unchanged."""
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().detach().float()
    return embeddings


def _concatenate_sparse_batches(batches: list[Any]) -> Any:
    """Concatenate per-batch sparse tensors along dim 0 (sparse tensors don't support `np.concatenate`)."""
    return torch.cat(batches, dim=0)


def _is_sparse_compatible_task(task_metadata: TaskMetadata) -> bool:
    """Whether `task_metadata`'s evaluator only calls `model.similarity(...)` on the raw embeddings.

    Such evaluators never index into the embeddings or hand them to sklearn/numpy directly, so they
    can consume the sparse tensors a SparseEncoder produces natively. Every other task needs dense
    embeddings: torch sparse COO tensors support neither generic indexing (e.g. Classification's
    `embeddings[idxs]`) nor `numpy.asarray()` conversion (e.g. Clustering handing embeddings to
    sklearn's `KMeans`).

    Gated on `simplified_task_type` rather than the raw, modality-specific `type` so that e.g. a
    multimodal sparse encoder's `Any2AnyRetrieval`/`VisionCentricQA` tasks are covered the same way
    as plain text `Retrieval`, without having to enumerate every modality-specific retrieval type.

    Reranking-family task types ("Reranking", "InstructionReranking", "AudioReranking") share the
    "retrieval" `simplified_task_type` with plain Retrieval, but their search path indexes into a
    cached embeddings tensor by position (see search_wrappers.py's `_rerank_documents`, called for
    a `top_ranked` candidate list) rather than only ever computing a single query-vs-corpus
    similarity matrix. Sparse COO tensors don't support that indexing either (confirmed on both CPU
    and MPS backends), so these are excluded despite sharing the "retrieval" grouping.
    """
    return (
        task_metadata.simplified_task_type == "retrieval"
        and "Reranking" not in task_metadata.type
    )


def _postprocess_sparse_embeddings(embeddings: Any) -> Any:
    """Densify a batch's embeddings if it's a sparse torch tensor, then move to CPU float32.

    `.to_dense()` keeps the tensor on its original device, but most non-retrieval evaluators
    (sklearn, numpy) can only consume CPU tensors/arrays, so this also applies the same CPU/float32
    normalization as `_postprocess_dense_embeddings`.
    """
    if isinstance(embeddings, torch.Tensor) and embeddings.is_sparse:
        embeddings = embeddings.to_dense()
    return _postprocess_dense_embeddings(embeddings)


def _encode_batches(
    inputs: DataLoader[BatchedInput],
    *,
    is_multimodal: bool,
    encode_function: Callable[..., Any],
    prompt: str | None,
    modalities: list[Modalities],
    postprocess_batch: Callable[[Any], Any] | None = None,
    concatenate_batches: Callable[[list[Any]], Any] | None = None,
    **kwargs: Any,
) -> Array:
    """Encode `inputs` with `encode_function`, handling the multimodal vs text-only cases.

    For multimodal inputs (as detected by `_setup_modality_collator`), each batch is converted to
    per-sample modality dicts and encoded separately; the per-batch outputs are combined with
    `concatenate_batches` (default: `np.concatenate`), after each one is first passed through
    `postprocess_batch` (default: identity). For text-only inputs, all sentences are collected up
    front and encoded in a single call.

    Args:
        inputs: The inputs to encode.
        is_multimodal: Whether `inputs` exposes image/audio/video features (see `_setup_modality_collator`).
        encode_function: The (bound) encode method to call, e.g. `model.encode_query`.
        prompt: The prompt text to pass to `encode_function`, if any.
        modalities: The modalities the model supports, used to build per-sample dicts for multimodal inputs.
        postprocess_batch: Optional hook applied to each batch's raw output.
        concatenate_batches: Optional hook used to combine per-batch outputs for multimodal inputs.
        **kwargs: Additional arguments to pass to `encode_function`.

    Returns:
        The encoded inputs.
    """
    postprocess = postprocess_batch or (lambda embeddings: embeddings)
    concatenate = concatenate_batches or (
        lambda batches: np.concatenate(batches, axis=0)
    )

    if is_multimodal:
        all_embeddings = []
        for batch in tqdm(inputs, desc="Building multimodal embeddings"):
            batched_input = _batch_to_modality_dicts(batch, modalities)
            embeddings = encode_function(batched_input, prompt=prompt, **kwargs)
            all_embeddings.append(postprocess(embeddings))
        return cast("Array", concatenate(all_embeddings))

    sentences = [text for batch in inputs for text in batch["text"]]
    embeddings = encode_function(sentences, prompt=prompt, **kwargs)
    return cast("Array", postprocess(embeddings))


class SentenceTransformerEncoderWrapper(AbsEncoder):
    """Wrapper for SentenceTransformer models.

    Supports both text-only and multimodal (text + image + audio + video)
    inputs. When the input dataset exposes image/audio/video features, the
    encode method attaches the matching collator and feeds the model per-sample
    modality dicts; otherwise it falls back to the text-only fast path that
    uses ``encode_query``/``encode_document`` where available.
    """

    mteb_model_meta: ModelMeta

    def __init__(  # noqa: PLR0913
        self,
        model: str | SentenceTransformer,
        revision: str | None = None,
        device: str | None = None,
        model_prompts: dict[str, str] | None = None,
        *,
        embed_dim: int | None = None,
        fps: float | None = None,
        max_frames: int | None = None,
        num_frames: int | None = None,
        target_sampling_rate: int | None = None,
        max_samples: int | None = None,
        **kwargs: Any,
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
            fps: Target frames per second for video sampling (multimodal inputs only).
            max_frames: Safety cap on frames per video for FPS mode (multimodal inputs only).
            num_frames: If set, use fixed-sample mode instead of FPS-based (multimodal inputs only).
            target_sampling_rate: Sampling rate to resample audio to (multimodal inputs only). Defaults to 16000 when an audio/video collator is applied.
            max_samples: Maximum number of audio samples to keep (multimodal inputs only).
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
                    loader=type(self),
                )
            )
        else:
            self.model = model
            self.mteb_model_meta = ModelMeta.from_sentence_transformer_model(self.model)

        self.model_prompts = _resolve_model_prompts(self.model, model_prompts)

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

        self.fps = fps
        self.max_frames = max_frames
        self.num_frames = num_frames
        self.target_sampling_rate = target_sampling_rate
        self.max_samples = max_samples

    def similarity(self, embeddings1: Array, embeddings2: Array) -> Array:
        """Compute the similarity between two collections of embeddings."""
        if hasattr(self.model, "similarity") and callable(self.model.similarity):
            return cast("Array", self.model.similarity(embeddings1, embeddings2))
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
        if "precision" in kwargs:
            existing_experiment_kwargs = self.mteb_model_meta.experiment_kwargs
            output_dtype = OutputDType.from_str(kwargs["precision"])
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

        prompt = _resolve_prompt(self.model_prompts, task_metadata, prompt_type)

        is_multimodal = _setup_modality_collator(
            inputs,
            fps=self.fps,
            max_frames=self.max_frames,
            num_frames=self.num_frames,
            target_sampling_rate=self.target_sampling_rate,
            max_samples=self.max_samples,
        )
        from sentence_transformers import __version__ as st_version

        has_query_encode = (
            Version(st_version).release
            >= Version(SENTENCE_TRANSFORMERS_QUERY_ENCODE_VERSION).release
        )
        encode_function = _select_encode_function(
            self.model, prompt_type, has_query_encode=has_query_encode
        )

        return _encode_batches(
            inputs,
            is_multimodal=is_multimodal,
            encode_function=encode_function,
            prompt=prompt,
            modalities=self.mteb_model_meta.modalities,
            postprocess_batch=_postprocess_dense_embeddings,
            **kwargs,
        )


class SentenceTransformerMultimodalEncoderWrapper(SentenceTransformerEncoderWrapper):
    """Backwards-compatible alias for `SentenceTransformerEncoderWrapper`.

    The base wrapper now auto-detects multimodal inputs, so this subclass is
    kept only to avoid breaking existing ``loader=...`` references.
    """

    @deprecated(
        "This wrapper is deprecated. Use `SentenceTransformerMultimodalEncoderWrapper` for using processing multimodal inputs.",
    )
    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)


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

    def __init__(  # noqa: PLR0913
        self,
        model: CrossEncoder | str,
        revision: str | None = None,
        device: str | None = None,
        query_prefix: str = "",
        passage_prefix: str = "",
        *,
        fps: float | None = None,
        max_frames: int | None = None,
        num_frames: int | None = None,
        target_sampling_rate: int | None = None,
        max_samples: int | None = None,
        **kwargs: Any,
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
        self.fps = fps
        self.max_frames = max_frames
        self.num_frames = num_frames
        self.target_sampling_rate = target_sampling_rate
        self.max_samples = max_samples

    def _collect_inputs(
        self,
        loader: DataLoader[BatchedInput],
        prefix: str,
    ) -> list[Any]:
        """Return a list of items to feed to the cross-encoder.

        For text-only inputs this is a list of prefix-prepended strings; for
        multimodal inputs it is a list of per-sample modality dicts.
        """
        is_multimodal = _setup_modality_collator(
            loader,
            fps=self.fps,
            max_frames=self.max_frames,
            num_frames=self.num_frames,
            target_sampling_rate=self.target_sampling_rate,
            max_samples=self.max_samples,
        )
        if not is_multimodal:
            return [prefix + text for batch in loader for text in batch["text"]]

        items: list[dict[str, Any]] = []
        for batch in tqdm(loader, desc="Collecting multimodal inputs"):
            for sample in _batch_to_modality_dicts(
                batch,
                self.mteb_model_meta.modalities,
            ):
                if prefix and "text" in sample:
                    sample["text"] = prefix + sample["text"]
                items.append(sample)
        return items

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
        queries = self._collect_inputs(inputs1, self.query_prefix)
        corpus = self._collect_inputs(inputs2, self.passage_prefix)

        return cast(
            "Array",
            self.model.predict(
                list(zip(queries, corpus)),
                **kwargs,
            ),
        )


class SparseEncoderWrapper(AbsEncoder):
    """Wrapper for sentence-transformers `SparseEncoder` models.

    Supports both text-only and multimodal (text + image + audio + video) inputs, following the
    same auto-detection pattern as `SentenceTransformerEncoderWrapper`: when the input dataset
    exposes image/audio/video features, the encode method attaches the matching collator and feeds
    the model per-sample modality dicts; otherwise it uses the text-only fast path.

    Args:
        model: The SparseEncoder model to use. Can be a string (model name) or a SparseEncoder model.
        revision: The revision of the model to use.
        device: The device used to load the model.
        model_prompts: A dictionary mapping task names to prompt names. See
            `SentenceTransformerEncoderWrapper` for the order of priority used to select a prompt.
        fps: Target frames per second for video sampling (multimodal inputs only).
        max_frames: Safety cap on frames per video for FPS mode (multimodal inputs only).
        num_frames: If set, use fixed-sample mode instead of FPS-based (multimodal inputs only).
        target_sampling_rate: Sampling rate to resample audio to (multimodal inputs only). Defaults to 16000 when an audio/video collator is applied.
        max_samples: Maximum number of audio samples to keep (multimodal inputs only).
        **kwargs: Additional arguments to pass to the SparseEncoder model.
    """

    mteb_model_meta: ModelMeta

    def __init__(
        self,
        model: str | SparseEncoder,
        revision: str | None = None,
        *,
        device: str | None = None,
        model_prompts: dict[str, str] | None = None,
        fps: float | None = None,
        max_frames: int | None = None,
        num_frames: int | None = None,
        target_sampling_rate: int | None = None,
        max_samples: int | None = None,
        **kwargs: Any,
    ) -> None:
        import sentence_transformers

        if (
            Version(sentence_transformers.__version__).release
            < Version(SENTENCE_TRANSFORMERS_QUERY_ENCODE_VERSION).release
        ):
            raise ImportError(
                f"sentence-transformers version must be >= {SENTENCE_TRANSFORMERS_QUERY_ENCODE_VERSION} to load a SparseEncoder model."
            )
        from sentence_transformers.sparse_encoder import SparseEncoder

        if isinstance(model, str):
            self.model = SparseEncoder(
                model, revision=revision, device=device, **kwargs
            )
            self.mteb_model_meta = ModelMeta.create_empty(
                overwrites=dict(
                    name=model,
                    revision=revision,
                    loader=type(self),
                    model_type=["sparse"],
                )
            )
        else:
            self.model = model
            self.mteb_model_meta = ModelMeta.from_sparse_encoder_model(self.model)

        self.model_prompts = _resolve_model_prompts(self.model, model_prompts)

        self.fps = fps
        self.max_frames = max_frames
        self.num_frames = num_frames
        self.target_sampling_rate = target_sampling_rate
        self.max_samples = max_samples

    def similarity(
        self,
        embeddings1: Array,
        embeddings2: Array,
    ) -> Array:
        """Compute similarity between sparse query_embeddings and corpus_embeddings.

        Args:
            embeddings1: Sparse COO tensor of shape (num_queries, dim).
            embeddings2: Tensor of shape (num_corpus, dim).

        Returns:
            Similarity matrix of shape (num_queries, num_corpus).
        """
        return cast("Array", self.model.similarity(embeddings1, embeddings2))

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
        """Encodes the given sentences using the sparse encoder.

        Args:
            inputs: The sentences (or, for multimodal models, image/audio/video samples) to encode.
            task_metadata: The metadata of the task. Used to determine which prompt to use
                from `model_prompts`.
            hf_split: Split of current task.
            hf_subset: Subset of current task.
            prompt_type: The name type of prompt (query or document).
            **kwargs: Additional arguments to pass to the encoder.

        Returns:
            The encoded inputs. Kept as sparse tensors for task types whose evaluators only need
            `similarity()` (see `_is_sparse_compatible_task`); densified for every other task type,
            since most evaluators need to index into or numpy-convert the embeddings.
        """
        prompt = _resolve_prompt(self.model_prompts, task_metadata, prompt_type)

        is_multimodal = _setup_modality_collator(
            inputs,
            fps=self.fps,
            max_frames=self.max_frames,
            num_frames=self.num_frames,
            target_sampling_rate=self.target_sampling_rate,
            max_samples=self.max_samples,
        )
        encode_function = _select_encode_function(self.model, prompt_type)

        postprocess_batch = (
            None
            if _is_sparse_compatible_task(task_metadata)
            else _postprocess_sparse_embeddings
        )

        return _encode_batches(
            inputs,
            is_multimodal=is_multimodal,
            encode_function=encode_function,
            prompt=prompt,
            modalities=self.mteb_model_meta.modalities,
            postprocess_batch=postprocess_batch,
            concatenate_batches=_concatenate_sparse_batches,
            **kwargs,
        )
