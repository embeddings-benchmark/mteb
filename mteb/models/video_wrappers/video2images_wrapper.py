from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.models.model_meta import ModelMeta
    from mteb.models.models_protocols import EncoderProtocol
    from mteb.types import Array, BatchedInput, PromptType

DEFAULT_NUM_FRAMES = 8


def video2images_model_meta(
    meta: ModelMeta,
    *,
    num_frames: int | None = None,
    fps: float | None = None,
    max_frames: int | None = None,
) -> ModelMeta:
    """Meta of an image model evaluated on video through frame sampling and mean pooling.

    Adds ``"video"`` to the modalities and records the sampling settings in
    ``experiment_kwargs`` so results are stored apart from those of native video models.
    """
    experiment_kwargs = dict(meta.experiment_kwargs or {})
    experiment_kwargs["video_frame_pooling"] = "mean"
    if num_frames is not None:
        experiment_kwargs["video_num_frames"] = num_frames
    if fps is not None:
        experiment_kwargs["video_fps"] = fps
    if max_frames is not None:
        experiment_kwargs["video_max_frames"] = max_frames
    modalities = list(meta.modalities)
    if "video" not in modalities:
        modalities.append("video")
    return meta.model_copy(
        update={"modalities": modalities, "experiment_kwargs": experiment_kwargs}
    )


class Video2ImagesWrapper:
    """Runs an image encoder on video tasks by encoding sampled frames and mean-pooling them.

    Frames are sampled across each clip, encoded independently as images by the wrapped
    model, and averaged into one video embedding. This is the protocol used to report
    CLIP-style models on video retrieval in e.g. CLIP4Clip and ChinaOpen.

    Frames are sampled either as a fixed number per clip (``num_frames``, the default) or at a
    rate (``fps``, optionally capped by ``max_frames``), matching the video models in MTEB.

    ``mteb.evaluate`` applies this wrapper automatically when an image model is run on a
    video task, so it only needs to be used directly for custom pipelines.

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
                ``DEFAULT_NUM_FRAMES`` when ``fps`` is not given.
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
        if num_frames is not None and fps is not None:
            raise ValueError("Use either `num_frames` or `fps`, not both.")
        if num_frames is None and fps is None:
            num_frames = DEFAULT_NUM_FRAMES
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
                **kwargs,
            )

        other_modalities = sorted(set(features) & {"text", "image", "audio"})
        if other_modalities:
            raise NotImplementedError(
                f"{type(self).__name__} only handles video-only inputs, "
                f"got video together with {other_modalities}."
            )

        import torch
        from datasets import Dataset, Features
        from datasets import Image as ImageFeature
        from torchvision.transforms.functional import to_pil_image
        from tqdm.auto import tqdm

        from mteb._create_dataloaders import create_dataloader
        from mteb.models.modality_collators import FramesCollator

        inputs.collate_fn = FramesCollator(
            num_frames=self.num_frames, fps=self.fps, max_frames=self.max_frames
        )
        image_task_metadata = _video_to_image_metadata(task_metadata)
        show_progress_bar = kwargs.pop("show_progress_bar", True)

        video_embeddings: list[torch.Tensor] = []
        for batch in tqdm(inputs, desc="Video Encoding", disable=not show_progress_bar):
            videos = batch["video"]
            frame_counts = [len(video) for video in videos]
            images = [to_pil_image(frame) for video in videos for frame in video]
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


def _video_to_image_metadata(task_metadata: TaskMetadata) -> TaskMetadata:
    modalities = [m for m in task_metadata.modalities if m != "video"]
    if "image" not in modalities:
        modalities.append("image")
    update: dict[str, Any] = {"modalities": modalities}
    if task_metadata.category is not None:
        update["category"] = task_metadata.category.replace("v", "i")
    return task_metadata.model_copy(update=update)
