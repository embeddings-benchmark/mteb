from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.models.model_meta import ModelMeta
    from mteb.models.models_protocols import EncoderProtocol
    from mteb.types import Array, BatchedInput, PromptType

DEFAULT_NUM_FRAMES = 8


class VideoFramesWrapper:
    """Runs an image encoder on video tasks by encoding sampled frames and mean-pooling them.

    Frames are sampled uniformly across each clip, encoded independently as images by the
    wrapped model, and averaged into one video embedding. This is the protocol used to report
    CLIP-style models on video retrieval in e.g. CLIP4Clip and ChinaOpen. The frame count and
    pooling are recorded in ``ModelMeta.experiment_kwargs`` so results are not confused with
    those of native video models.

    Examples:
        >>> import mteb
        >>> from mteb.models import VideoFramesWrapper
        >>> model = mteb.get_model("openai/clip-vit-base-patch32")
        >>> video_model = VideoFramesWrapper(model, num_frames=8)
        >>> task = mteb.get_task("MSRVTTT2V")
        >>> mteb.evaluate(video_model, task)
    """

    def __init__(
        self,
        model: EncoderProtocol,
        *,
        num_frames: int = DEFAULT_NUM_FRAMES,
    ) -> None:
        """Wrap an image encoder so it can be evaluated on video tasks.

        Args:
            model: An encoder whose ``mteb_model_meta.modalities`` includes ``"image"``.
            num_frames: Number of frames sampled uniformly from each video.
        """
        meta = model.mteb_model_meta
        if meta is None or "image" not in meta.modalities:
            raise ValueError(
                f"{type(self).__name__} requires a model that supports the 'image' modality, "
                f"got modalities={meta.modalities if meta else None}."
            )
        if num_frames < 1:
            raise ValueError(f"`num_frames` must be at least 1, got {num_frames}.")

        self.model = model
        self.num_frames = num_frames

        experiment_kwargs = dict(meta.experiment_kwargs or {})
        experiment_kwargs["video_num_frames"] = num_frames
        experiment_kwargs["video_frame_pooling"] = "mean"
        modalities = list(meta.modalities)
        if "video" not in modalities:
            modalities.append("video")
        model.mteb_model_meta = meta.model_copy(  # type: ignore[misc]
            update={"modalities": modalities, "experiment_kwargs": experiment_kwargs}
        )

    @property
    def mteb_model_meta(self) -> ModelMeta | None:
        """The wrapped model meta data."""
        return self.model.mteb_model_meta

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
        from tqdm import tqdm

        from mteb._create_dataloaders import create_dataloader
        from mteb.models.modality_collators import FramesCollator

        inputs.collate_fn = FramesCollator(num_frames=self.num_frames)
        image_task_metadata = _video_to_image_metadata(task_metadata)
        show_progress_bar = kwargs.pop("show_progress_bar", True)

        video_embeddings = []
        for batch in tqdm(inputs, desc="Video Encoding", disable=not show_progress_bar):
            videos = batch["video"]
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
            video_embeddings.append(
                frame_embeddings.view(len(videos), self.num_frames, -1).mean(dim=1)
            )
        return torch.cat(video_embeddings)

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
