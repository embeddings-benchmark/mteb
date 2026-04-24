from __future__ import annotations

import logging
import os
import pathlib
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

    EBind's processor expects file paths, so MTEB's decoded inputs (PIL
    images, frame tensors, audio arrays) are written to temporary files
    before being passed through the unified processor.
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

    @staticmethod
    def _save_image(image: Any) -> str:
        """Write a PIL Image to a temporary PNG file."""
        fd, path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        if image.mode not in {"RGB", "RGBA", "L", "LA", "P"}:
            image = image.convert("RGB")
        image.save(path)
        return path

    @staticmethod
    def _save_video(frames: torch.Tensor) -> str:
        """Write a ``(T, C, H, W)`` uint8 tensor to a temporary MP4 file."""
        import torchvision

        fd, path = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)
        torchvision.io.write_video(
            path, frames.permute(0, 2, 3, 1).contiguous(), fps=24
        )
        return path

    @staticmethod
    def _save_audio(audio_data: Any) -> str:
        """Write an audio array or AudioInputItem dict to a temporary WAV file."""
        import soundfile as sf

        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        if isinstance(audio_data, dict) and "array" in audio_data:
            sf.write(path, audio_data["array"], audio_data["sampling_rate"])
        else:
            sf.write(path, audio_data, 16_000)
        return path

    @torch.inference_mode()
    def _encode_batch(self, batch: BatchedInput) -> torch.Tensor:
        """Encode all modalities via the unified processor and a single forward pass.

        When multiple modalities are present (e.g. video+audio from the same
        clip), embeddings are fused by element-wise addition and renormalised.
        """
        temp_files: list[str] = []
        try:
            processor_inputs: dict[str, list[str]] = {}

            if batch.get("text"):
                processor_inputs["text"] = batch["text"]

            if batch.get("image"):
                paths = [self._save_image(img) for img in batch["image"]]
                temp_files.extend(paths)
                processor_inputs["image"] = paths

            if batch.get("video"):
                paths = [self._save_video(v) for v in batch["video"]]
                temp_files.extend(paths)
                processor_inputs["video"] = paths

            if batch.get("audio"):
                paths = [self._save_audio(a) for a in batch["audio"]]
                temp_files.extend(paths)
                processor_inputs["audio"] = paths

            if not processor_inputs:
                raise ValueError(
                    f"No supported modality found in batch: {list(batch.keys())}"
                )

            processed = self.processor(processor_inputs, return_tensors="pt")
            outputs = self.model.forward(**processed)

            # Fuse by addition when multiple modalities are present
            embeddings = None
            for emb in outputs.values():
                embeddings = emb if embeddings is None else embeddings + emb

            return torch.nn.functional.normalize(embeddings, p=2, dim=-1)

        finally:
            for f in temp_files:
                try:
                    pathlib.Path(f).unlink()
                except OSError:
                    pass

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
