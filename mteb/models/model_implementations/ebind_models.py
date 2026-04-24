from __future__ import annotations

import logging
import tempfile
from typing import TYPE_CHECKING, Any

import torch
from tqdm.auto import tqdm

from mteb._create_dataloaders import AudioCollator, VideoCollator
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from mteb import TaskMetadata
    from mteb.types import Array, BatchedInput, PromptType


class EBindWrapper(AbsEncoder):
    """MTEB wrapper for EBind multi-modal embedding model.

    EBind projects image, video, audio, and text into a shared 1024-dim
    embedding space using Perception Encoder, ImageBind, and Uni3D backbones
    with learned projection heads.

    EBind's processors expect file paths. This wrapper bridges to MTEB's
    decoded inputs: image transforms are applied directly on PIL images,
    video frames are resized/normalised as tensors, and audio arrays are
    written to temporary wav files for ImageBind's mel-spectrogram pipeline.
    """

    def __init__(
        self,
        model_name: str,
        revision: str | None = None,
        device: str | None = None,
        fps: float | None = None,
        max_frames: int | None = None,
        num_frames: int | None = 8,
        **kwargs: Any,
    ) -> None:
        from ebind import EBindModel, EBindProcessor

        self.device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self.fps = fps
        self.max_frames = max_frames
        self.num_frames = num_frames

        self.model = EBindModel.from_pretrained(model_name, revision=revision)
        self.model.to(self.device).eval()

        self.processor = EBindProcessor.from_pretrained(model_name, revision=revision)
        self.processor = self.processor.to(self.device)

        self._image_transform = self.processor.processors["image"].transform
        self._image_size = self.processor.processors["image"].model_image_size

    def _process_audio_batch(self, audio_items: list) -> torch.Tensor:
        """Process a batch of audio items via temp WAV files (IBAudioProcessor requires paths)."""
        import soundfile as sf

        audio_tensors = []
        for item in audio_items:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
                sf.write(tmp.name, item["array"], item["sampling_rate"])
                audio_tensors.append(self.processor.processors["audio"](tmp.name))
        return torch.stack(audio_tensors).to(self.device)

    def _process_video(self, raw_frames: torch.Tensor) -> torch.Tensor:
        """Resize and normalise raw frame tensors for the PE vision encoder."""
        from torchvision.transforms.functional import (
            InterpolationMode,
            normalize,
            resize,
        )

        processed = resize(
            raw_frames,
            [self._image_size, self._image_size],
            interpolation=InterpolationMode.BICUBIC,
        )
        processed = processed.float() / 255.0
        return normalize(processed, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    @torch.inference_mode()
    def _encode_batch(self, batch: BatchedInput) -> torch.Tensor:
        """Encode all modalities in a batch, fusing by addition when mixed."""
        embeddings = None

        if batch.get("text"):
            processed = self.processor({"text": batch["text"]}, return_tensors="pt")
            text_emb = self.model.forward(**processed)["text"]
            embeddings = text_emb

        if batch.get("image"):
            img_tensors = torch.stack(
                [self._image_transform(img) for img in batch["image"]]
            ).to(self.device)
            image_emb = self.model.forward(image=img_tensors)["image"]
            embeddings = image_emb if embeddings is None else embeddings + image_emb

        if batch.get("video"):
            processed = [self._process_video(v) for v in batch["video"]]
            # Fixed frame count: stack into one batched forward pass
            # Variable frame count (FPS mode): encode each video individually
            if len({v.shape[0] for v in processed}) == 1:
                stacked = torch.stack(processed).to(self.device)
                video_emb = self.model.forward(video=stacked)["video"]
            else:
                logger.warning(
                    "Variable frame counts in batch — falling back to per-video encoding. "
                    "Use fixed num_frames (default 8) instead of fps for batched processing."
                )
                video_emb = torch.cat(
                    [
                        self.model.forward(video=v.unsqueeze(0).to(self.device))[
                            "video"
                        ]
                        for v in processed
                    ]
                )
            embeddings = video_emb if embeddings is None else embeddings + video_emb

        if batch.get("audio"):
            audio_tensors = self._process_audio_batch(batch["audio"])
            audio_emb = self.model.forward(audio=audio_tensors)["audio"]
            embeddings = audio_emb if embeddings is None else embeddings + audio_emb

        if embeddings is None:
            raise ValueError(
                f"No supported modality found in batch: {list(batch.keys())}"
            )

        return torch.nn.functional.normalize(embeddings, p=2, dim=-1)

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
        features = inputs.dataset.features
        has_video = "video" in features
        has_audio = "audio" in features

        if has_video:
            inputs.collate_fn = VideoCollator(
                target_sampling_rate=16_000,
                fps=self.fps,
                max_frames=self.max_frames,
                num_frames=self.num_frames,
            )
        elif has_audio:
            inputs.collate_fn = AudioCollator(
                target_sampling_rate=16_000,
            )

        all_embeddings: list[torch.Tensor] = []
        for batch in tqdm(inputs, desc="Encoding"):
            emb = self._encode_batch(batch)
            all_embeddings.append(emb.cpu())

        return torch.cat(all_embeddings, dim=0).float()


_EBIND_CITATION = r"""
@misc{broadbent2025ebindpracticalapproachspace,
      title={{EBind}: a practical approach to space binding},
      author={Jim Broadbent and Felix Cohen and Frederik Hvilshøj and Eric Landau and Eren Sasoglu},
      year={2025},
      eprint={2511.14229},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2511.14229},
}
"""

_EBIND_COMMON = dict(
    languages=["eng-Latn"],
    release_date="2025-11-19",
    max_tokens=512,
    n_embedding_parameters=50_593_792,  # PE text encoder: 49408 vocab * 1024 hidden
    license="cc-by-nc-sa-4.0",
    open_weights=True,
    public_training_code="https://github.com/encord-team/ebind",
    public_training_data=None,
    framework=["PyTorch", "safetensors"],
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets=set(),
    citation=_EBIND_CITATION,
    extra_requirements_groups=["ebind"],
)

# n_parameters is the total loaded at runtime (backbone encoders + projectors).
# Safetensors only stores 8.4M projection params; backbones load from separate repos.

ebind_full = ModelMeta(
    loader=EBindWrapper,
    name="encord-team/ebind-full",
    revision="482831d6cb97d3ffc970f933c614a03aa9891416",
    n_parameters=1_790_000_000,
    memory_usage_mb=6828,
    embed_dim=1024,
    modalities=["text", "image", "audio", "video"],
    reference="https://huggingface.co/encord-team/ebind-full",
    **_EBIND_COMMON,
)

ebind_audio_vision = ModelMeta(
    loader=EBindWrapper,
    name="encord-team/ebind-audio-vision",
    revision="c16c21588ea034a2198f9b5e083ae7805434c198",
    n_parameters=764_200_000,
    memory_usage_mb=2915,
    embed_dim=1024,
    modalities=["text", "image", "audio", "video"],
    reference="https://huggingface.co/encord-team/ebind-audio-vision",
    **_EBIND_COMMON,
)

ebind_points_vision = ModelMeta(
    loader=EBindWrapper,
    name="encord-team/ebind-points-vision",
    revision="9941fa57df50c13e599e608b47da1042c76c222b",
    n_parameters=1_694_200_000,
    memory_usage_mb=6463,
    embed_dim=1024,
    modalities=["text", "image", "video"],
    reference="https://huggingface.co/encord-team/ebind-points-vision",
    **_EBIND_COMMON,
)
