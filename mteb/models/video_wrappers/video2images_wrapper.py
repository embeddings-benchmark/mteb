from __future__ import annotations

import io
from typing import TYPE_CHECKING, Any

from datasets import Dataset, Features
from datasets import Image as ImageFeature
from tqdm.auto import tqdm

from mteb._create_dataloaders import create_dataloader
from mteb.models.modality_collators import FramesCollator

if TYPE_CHECKING:
    import torch
    from torch.utils.data import DataLoader

    from mteb.abstasks.abstask import AbsTask
    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.models.model_meta import ModelMeta
    from mteb.models.models_protocols import EncoderProtocol
    from mteb.types import Array, BatchedInput, PromptType

VIDEO_FRAME_POOLING_KEY = "video_frame_pooling"


def is_video2images_meta(meta: ModelMeta) -> bool:
    """Whether `meta` belongs to a model wrapped in `Video2ImagesWrapper`.

    Wrapped models are recognised by the frame-pooling entry that `video2images_model_meta`
    always records in ``experiment_kwargs``. A model that set the same key itself would be
    treated as wrapped.
    """
    return VIDEO_FRAME_POOLING_KEY in (meta.experiment_kwargs or {})


def video2images_model_meta(
    meta: ModelMeta,
    *,
    num_frames: int | None = None,
    fps: float | None = None,
    max_frames: int | None = None,
) -> ModelMeta:
    """Meta of an image model evaluated on video through frame sampling and mean pooling.

    Adds ``"video"`` to the modalities and records the sampling in ``experiment_kwargs``, so the
    results are stored as an experiment and stay distinguishable from native video models.
    """
    modalities = list(meta.modalities)
    if "video" not in modalities:
        modalities.append("video")
    experiment_kwargs = dict(meta.experiment_kwargs or {})
    experiment_kwargs[VIDEO_FRAME_POOLING_KEY] = "mean"
    if num_frames is not None:
        experiment_kwargs["video_num_frames"] = num_frames
    if fps is not None:
        experiment_kwargs["video_fps"] = fps
    if max_frames is not None:
        experiment_kwargs["video_max_frames"] = max_frames
    return meta.model_copy(
        update={
            "modalities": modalities,
            "experiment_kwargs": experiment_kwargs,
        }
    )


_CATEGORY_MODALITIES: dict[str, str] = {
    "t": "text",
    "i": "image",
    "a": "audio",
    "v": "video",
}


def _input_modality_groups(task: AbsTask) -> list[set[str]]:
    """Groups of modalities that are encoded together in one input of `task`.

    Retrieval tasks have a query and a document side. For other task types every side of the
    category that names modalities is an input (e.g. both sides of ``va2va`` pair
    classification, the left side of ``va2c`` classification); sides naming a label, such as the
    ``c`` of ``v2c``, are not. The modalities listed in the task metadata win over a stale
    category, so a side is only taken as far as the task actually has those modalities.
    """
    from mteb.abstasks.retrieval import AbsTaskRetrieval
    from mteb.types import PromptType

    metadata = task.metadata
    if isinstance(task, AbsTaskRetrieval):
        return [
            set(metadata.get_modalities(PromptType.query)),
            set(metadata.get_modalities(PromptType.document)),
        ]
    if metadata.category is None:
        return []
    sides = metadata.category.split("2")
    return [
        {_CATEGORY_MODALITIES[m] for m in side if m in _CATEGORY_MODALITIES}
        & set(metadata.modalities)
        for side in sides
    ]


def _has_mixed_video_input(task: AbsTask) -> bool:
    """Whether `task` combines video with another modality in a single input.

    `Video2ImagesWrapper` pools frames of video-only inputs, so such tasks would fail mid-run.
    """
    return any(
        "video" in group and group != {"video"}
        for group in _input_modality_groups(task)
    )


def _wrapper_can_run(meta: ModelMeta, task: AbsTask) -> bool:
    """Whether wrapping the model would let it run `task`: video added passes the modality check, no mixed video side."""
    from mteb.abstasks.retrieval import AbsTaskRetrieval
    from mteb.types import PromptType

    if _has_mixed_video_input(task):
        return False
    supported = set(meta.modalities) | {"video"}
    if isinstance(task, AbsTaskRetrieval):
        return bool(
            supported & set(task.metadata.get_modalities(PromptType.query))
        ) and bool(supported & set(task.metadata.get_modalities(PromptType.document)))
    return set(task.metadata.modalities).issubset(supported)


def video2images_hint(meta: ModelMeta, task: AbsTask) -> str:
    """A sentence pointing to `Video2ImagesWrapper`, or an empty string if wrapping would not help."""
    if (
        "image" not in meta.modalities
        or "video" in meta.modalities
        or "dense" not in meta.model_type
        or "video" not in task.metadata.modalities
        or not _wrapper_can_run(meta, task)
    ):
        return ""
    return (
        f" {meta.name} does not run on video and {task.metadata.name} contains video. "
        "If you wish to evaluate it on frames sampled from the videos, wrap it in "
        "`mteb.models.Video2ImagesWrapper` "
        "(see https://embeddings-benchmark.github.io/mteb/get_started/advanced_usage/image_models_on_video/)."
    )


class Video2ImagesWrapper:
    """Runs an image encoder on video tasks by encoding sampled frames and mean-pooling them.

    Frames are sampled across each clip, encoded independently as images by the wrapped
    model, and averaged into one video embedding. This is the protocol used to report
    CLIP-style models on video retrieval in e.g. CLIP4Clip and ChinaOpen.

    Frames are sampled either as a fixed number per clip (``num_frames``, the default) or at a
    rate (``fps``, optionally capped by ``max_frames``), matching the video models in MTEB.
    The sampling is recorded in ``experiment_kwargs``, so results are stored as an experiment.

    ``mteb.evaluate`` never applies this wrapper implicitly: wrap the model explicitly to run it
    on video tasks.

    Examples:
        >>> import mteb
        >>> from mteb.models import Video2ImagesWrapper
        >>> model = mteb.get_model("openai/clip-vit-base-patch32")
        >>> video_model = Video2ImagesWrapper(model, num_frames=8)
        >>> task = mteb.get_task("MSRVTTT2V")
        >>> mteb.evaluate(video_model, task)
    """

    def __init__(
        self,
        model: EncoderProtocol,
        *,
        num_frames: int | None = None,
        fps: float | None = None,
        max_frames: int | None = None,
    ) -> None:
        """Wrap an image encoder so it can be evaluated on video tasks.

        Args:
            model: An encoder whose ``mteb_model_meta.modalities`` includes ``"image"``.
            num_frames: Number of frames sampled uniformly from each video. Defaults to
                8 when ``fps`` is not given.
            fps: Sample frames at this rate instead of a fixed count. Cannot be combined
                with ``num_frames``.
            max_frames: Cap on the number of frames per video when sampling by ``fps``.
        """
        meta = model.mteb_model_meta
        if meta is None or "image" not in meta.modalities:
            raise ValueError(
                f"{type(self).__name__} requires a model that supports the 'image' modality, "
                f"got modalities={meta.modalities if meta else None}."
            )
        if "video" in meta.modalities:
            raise ValueError(
                f"{meta.name} already supports the 'video' modality, so wrapping it in "
                f"{type(self).__name__} would replace its native video path with mean-pooled "
                "frames and store the result as its regular video results. Evaluate it directly."
            )
        if "dense" not in meta.model_type:
            raise ValueError(
                f"{type(self).__name__} mean-pools one embedding per frame, which requires a "
                f"dense encoder, got model_type={meta.model_type}. Late-interaction models emit "
                "variable-length per-token embeddings and cross-encoders have no `encode`, so "
                "neither can be pooled this way."
            )
        if num_frames is not None and fps is not None:
            raise ValueError("Use either `num_frames` or `fps`, not both.")
        if num_frames is None and fps is None:
            num_frames = 8
        if num_frames is not None and num_frames < 1:
            raise ValueError(f"`num_frames` must be at least 1, got {num_frames}.")

        self.model = model
        self.num_frames = num_frames
        self.fps = fps
        self.max_frames = max_frames
        self.mteb_model_meta = video2images_model_meta(
            meta, num_frames=num_frames, fps=fps, max_frames=max_frames
        )

    def encode(
        self,
        inputs: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        show_progress_bar: bool = True,
        **kwargs: Any,
    ) -> Array:
        """Encode inputs, turning video rows into mean-pooled frame embeddings.

        Non-video inputs (e.g. text queries) are passed straight to the wrapped model.

        Args:
            inputs: Batch of inputs to encode.
            task_metadata: The metadata of the task.
            hf_split: Split of current task.
            hf_subset: Subset of current task.
            prompt_type: The name type of prompt (query or passage).
            show_progress_bar: Whether to show a progress bar over the videos.
            **kwargs: Additional arguments to pass to the wrapped encoder.

        Returns:
            One embedding per input row.
        """
        features = inputs.dataset.features  # type: ignore[attr-defined]
        if "video" not in features:
            return self.model.encode(
                inputs,
                task_metadata=task_metadata,
                hf_split=hf_split,
                hf_subset=hf_subset,
                prompt_type=prompt_type,
                show_progress_bar=show_progress_bar,
                **kwargs,
            )

        other_modalities = sorted(set(features) & {"text", "image", "audio"})
        if other_modalities:
            raise NotImplementedError(
                f"{type(self).__name__} only handles video-only inputs, "
                f"got video together with {other_modalities}."
            )

        import torch

        inputs.collate_fn = FramesCollator(
            num_frames=self.num_frames, fps=self.fps, max_frames=self.max_frames
        )
        image_task_metadata = _video_to_image_metadata(task_metadata)

        video_embeddings: list[torch.Tensor] = []
        for batch in tqdm(inputs, desc="Video Encoding", disable=not show_progress_bar):
            videos = batch["video"]
            frame_counts = [len(video) for video in videos]
            images = [_frame_to_bytes(frame) for video in videos for frame in video]
            image_loader = create_dataloader(
                Dataset.from_dict(
                    {"image": images}, features=Features({"image": ImageFeature()})
                ),
                task_metadata=image_task_metadata,
                prompt_type=prompt_type,
                batch_size=inputs.batch_size or 32,
            )
            frame_embeddings = torch.as_tensor(
                self.model.encode(
                    image_loader,
                    task_metadata=image_task_metadata,
                    hf_split=hf_split,
                    hf_subset=hf_subset,
                    prompt_type=prompt_type,
                    show_progress_bar=False,
                    **kwargs,
                )
            )
            video_embeddings.extend(
                chunk.mean(dim=0)
                for chunk in torch.split(frame_embeddings, frame_counts)
            )
        return torch.stack(video_embeddings)

    def similarity(self, embeddings1: Array, embeddings2: Array) -> Array:
        """Refer to [EncoderProtocol.similarity][mteb.models.EncoderProtocol.similarity] for more details."""
        return self.model.similarity(embeddings1, embeddings2)

    def similarity_pairwise(self, embeddings1: Array, embeddings2: Array) -> Array:
        """Refer to [EncoderProtocol.similarity_pairwise][mteb.models.EncoderProtocol.similarity_pairwise] for more details."""
        return self.model.similarity_pairwise(embeddings1, embeddings2)


def _frame_to_bytes(frame: torch.Tensor) -> dict[str, Any]:
    """Encode a frame as BMP bytes: lossless and much cheaper than the default PNG encoding."""
    from torchvision.transforms.functional import to_pil_image

    buffer = io.BytesIO()
    to_pil_image(frame).save(buffer, format="BMP")
    return {"bytes": buffer.getvalue(), "path": None}


def _video_to_image_metadata(task_metadata: TaskMetadata) -> TaskMetadata:
    modalities = [m for m in task_metadata.modalities if m != "video"]
    if "image" not in modalities:
        modalities.append("image")
    update: dict[str, Any] = {"modalities": modalities}
    if task_metadata.category is not None:
        update["category"] = task_metadata.category.replace("v", "i")
    return task_metadata.model_copy(update=update)
