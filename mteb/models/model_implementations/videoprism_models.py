from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from tqdm.auto import tqdm

from mteb.models.abs_encoder import AbsEncoder
from mteb.models.modality_collators import FramesCollator
from mteb.models.model_meta import ModelMeta, ScoringFunction

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput, PromptType

# Usage follows the transformers docs for VideoPrism:
# https://huggingface.co/docs/transformers/en/model_doc/videoprism
# The HF-format weights are not on main in any of the four repos yet, only the
# original Flax .npz, so the revisions pinned in the ModelMeta blocks below are
# commits from the open Hub PRs that carry the converted checkpoints.

# The LVT text tower has exactly 64 learned position embeddings and adds them
# elementwise, so a longer sequence raises a shape mismatch instead of being
# truncated internally. Every text batch is padded and truncated to this length.
MAX_TEXT_TOKENS = 64


def _resolve_device(device: str | int | torch.device | None) -> torch.device:
    if device is not None:
        # mteb's CLI may pass an int index, torch.device normalises it
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class VideoPrismVisionWrapper(AbsEncoder):
    """VideoPrism video-only encoders (Base, Large)."""

    def __init__(
        self,
        model_name: str,
        revision: str | None,
        device: str | int | torch.device | None = None,
        fps: float | None = None,
        max_frames: int | None = None,
        num_frames: int | None = 16,
        **kwargs: Any,
    ) -> None:
        from transformers import AutoVideoProcessor, VideoPrismVisionModel

        self.model_name = model_name
        self.fps = fps
        self.max_frames = max_frames
        self.num_frames = num_frames
        self.device = _resolve_device(device)
        self.dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.processor = AutoVideoProcessor.from_pretrained(
            model_name, revision=revision
        )
        self.model = VideoPrismVisionModel.from_pretrained(
            model_name, revision=revision, dtype=self.dtype
        ).to(self.device)
        self.model.eval()

    @torch.inference_mode()
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
        inputs.collate_fn = FramesCollator(
            fps=self.fps,
            max_frames=self.max_frames,
            num_frames=self.num_frames,
        )

        all_embeddings = []
        for batch in tqdm(inputs, disable=not show_progress_bar, desc="Video Encoding"):
            # Source clips vary in resolution, so they cannot be stacked before
            # preprocessing. The processor resizes each clip to 288x288, after
            # which they concatenate cleanly.
            processed = []
            for video in batch["video"]:
                clip = video.to(torch.uint8)
                processed.append(
                    self.processor(videos=[clip.numpy()], return_tensors="pt")[
                        "pixel_values_videos"
                    ]
                )
            pixel_values_videos = torch.cat(processed, dim=0).to(
                self.device, dtype=self.dtype
            )
            output = self.model(pixel_values_videos=pixel_values_videos)
            # No pooler head on the vision-only checkpoints. last_hidden_state is
            # (batch, num_frames * num_patches, hidden), so mean over the token
            # axis, matching how the vjepa2 wrapper pools.
            pooled = output.last_hidden_state.mean(dim=1)
            all_embeddings.append(pooled.float().cpu())
        return torch.cat(all_embeddings, dim=0).numpy()


class VideoPrismClipWrapper(AbsEncoder):
    """VideoPrism LVT video-text encoders (LVT Base, LVT Large)."""

    def __init__(
        self,
        model_name: str,
        revision: str | None,
        device: str | int | torch.device | None = None,
        fps: float | None = None,
        max_frames: int | None = None,
        num_frames: int | None = 16,
        **kwargs: Any,
    ) -> None:
        from transformers import AutoProcessor, VideoPrismClipModel

        self.model_name = model_name
        self.fps = fps
        self.max_frames = max_frames
        self.num_frames = num_frames
        self.device = _resolve_device(device)
        self.dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.processor = AutoProcessor.from_pretrained(model_name, revision=revision)
        self.model = VideoPrismClipModel.from_pretrained(
            model_name, revision=revision, dtype=self.dtype
        ).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def get_text_embeddings(
        self,
        texts: DataLoader[BatchedInput],
        show_progress_bar: bool = True,
        **kwargs: Any,
    ) -> torch.Tensor:
        all_embeddings = []
        for batch in tqdm(texts, disable=not show_progress_bar, desc="Text Encoding"):
            encoded = self.processor.tokenizer(
                batch["text"],
                padding="max_length",
                truncation=True,
                max_length=MAX_TEXT_TOKENS,
                return_tensors="pt",
            ).to(self.device)
            output = self.model.get_text_features(**encoded)
            all_embeddings.append(output.pooler_output.float().cpu())
        return torch.cat(all_embeddings, dim=0)

    @torch.inference_mode()
    def get_video_embeddings(
        self,
        videos: DataLoader[BatchedInput],
        show_progress_bar: bool = True,
        **kwargs: Any,
    ) -> torch.Tensor:
        all_embeddings = []
        for batch in tqdm(videos, disable=not show_progress_bar, desc="Video Encoding"):
            processed = []
            for video in batch["video"]:
                clip = video.to(torch.uint8)
                processed.append(
                    self.processor(videos=[clip.numpy()], return_tensors="pt")[
                        "pixel_values_videos"
                    ]
                )
            pixel_values_videos = torch.cat(processed, dim=0).to(
                self.device, dtype=self.dtype
            )
            output = self.model.get_video_features(
                pixel_values_videos=pixel_values_videos
            )
            # The video pooler returns (batch, 1, hidden) while the text pooler
            # returns (batch, hidden). Drop the singleton axis so the two towers
            # are directly comparable.
            all_embeddings.append(output.pooler_output.squeeze(1).float().cpu())
        return torch.cat(all_embeddings, dim=0)

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
        has_text = "text" in inputs.dataset.features
        has_video = "video" in inputs.dataset.features

        if has_video:
            inputs.collate_fn = FramesCollator(
                fps=self.fps,
                max_frames=self.max_frames,
                num_frames=self.num_frames,
            )

        text_embeddings = None
        video_embeddings = None

        if has_text:
            text_embeddings = self.get_text_embeddings(inputs, **kwargs)
        if has_video:
            video_embeddings = self.get_video_embeddings(inputs, **kwargs)

        if text_embeddings is not None and video_embeddings is not None:
            if len(text_embeddings) != len(video_embeddings):
                raise ValueError(
                    "The number of texts and videos must have the same length"
                )
            return text_embeddings + video_embeddings
        if text_embeddings is not None:
            return text_embeddings
        if video_embeddings is not None:
            return video_embeddings

        raise ValueError(
            f"No supported modality found in dataset features: {list(inputs.dataset.features.keys())}"
        )


_VIDEOPRISM_CITATION = """@inproceedings{zhao2024videoprism,
  title = {{VideoPrism}: A Foundational Visual Encoder for Video Understanding},
  author = {Long Zhao and Nitesh B. Gundavarapu and Liangzhe Yuan and Hao Zhou and Shen Yan and Jennifer J. Sun and Luke Friedman and Rui Qian and Tobias Weyand and Yue Zhao and Rachel Hornung and Florian Schroff and Ming-Hsuan Yang and David A. Ross and Huisheng Wang and Hartwig Adam and Mikhail Sirotenko and Ting Liu and Boqing Gong},
  booktitle = {International Conference on Machine Learning (ICML)},
  year = {2024}
}"""

_VIDEOPRISM_COMMON = dict(
    model_type=["dense"],
    languages=["eng-Latn"],
    license="apache-2.0",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["PyTorch", "Transformers"],
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    # Pretrained on WebLI, InternVid, VideoCC and WTS-70M, none of which are
    # mteb tasks.
    training_datasets=None,
    citation=_VIDEOPRISM_CITATION,
    # The video extra ships datasets and torchcodec but not torchvision or
    # pillow, which every transformers video processor imports.
    extra_requirements_groups=["image"],
)

videoprism_base_f16r288 = ModelMeta(
    loader=VideoPrismVisionWrapper,
    name="google/videoprism-base-f16r288",
    revision="2be2ef51d0d611d0fadf21a6d1470a4ad4af9249",
    release_date="2025-06-03",
    modalities=["video"],
    n_parameters=114_365_184,
    n_embedding_parameters=None,
    memory_usage_mb=436,
    embed_dim=768,
    max_tokens=None,
    reference="https://huggingface.co/google/videoprism-base-f16r288",
    loader_kwargs=dict(num_frames=16),
    **_VIDEOPRISM_COMMON,
)

videoprism_large_f8r288 = ModelMeta(
    loader=VideoPrismVisionWrapper,
    name="google/videoprism-large-f8r288",
    revision="2e20e540e1cb89e734e2f2477590b7f8197250ca",
    release_date="2025-06-03",
    modalities=["video"],
    n_parameters=353_965_056,
    n_embedding_parameters=None,
    memory_usage_mb=1350,
    embed_dim=1024,
    max_tokens=None,
    reference="https://huggingface.co/google/videoprism-large-f8r288",
    loader_kwargs=dict(num_frames=8),
    **_VIDEOPRISM_COMMON,
)

videoprism_lvt_base_f16r288 = ModelMeta(
    loader=VideoPrismClipWrapper,
    name="google/videoprism-lvt-base-f16r288",
    revision="fb6de9f0eb7bc285be86bdca1cf7daa3e3ef51ff",
    release_date="2025-07-16",
    modalities=["video", "text"],
    n_parameters=247_623_424,
    n_embedding_parameters=24_576_000,
    memory_usage_mb=945,
    embed_dim=768,
    max_tokens=MAX_TEXT_TOKENS,
    reference="https://huggingface.co/google/videoprism-lvt-base-f16r288",
    loader_kwargs=dict(num_frames=16),
    **_VIDEOPRISM_COMMON,
)

videoprism_lvt_large_f8r288 = ModelMeta(
    loader=VideoPrismClipWrapper,
    name="google/videoprism-lvt-large-f8r288",
    revision="03900b4ecc0b278b30abdc5953c575bb5f083865",
    release_date="2025-07-16",
    modalities=["video", "text"],
    n_parameters=579_877_120,
    n_embedding_parameters=32_768_000,
    memory_usage_mb=2212,
    embed_dim=1024,
    max_tokens=MAX_TEXT_TOKENS,
    reference="https://huggingface.co/google/videoprism-lvt-large-f8r288",
    loader_kwargs=dict(num_frames=8),
    **_VIDEOPRISM_COMMON,
)
