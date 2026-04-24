from __future__ import annotations

import os
import pathlib
import tempfile
from typing import TYPE_CHECKING, Any

import torch
from tqdm.auto import tqdm
from transformers import AutoModel, AutoProcessor

from mteb._create_dataloaders import VideoCollator
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput, PromptType


class OmniVinciWrapper(AbsEncoder):
    """MTEB wrapper for NVIDIA OmniVinci.

    Uses last-token pooling over the final hidden states for embeddings.
    VILA's processor consumes file paths for video/audio, so MTEB's decoded
    frame tensors and audio arrays are written to temporary MP4/WAV files.
    """

    AUDIO_SAMPLING_RATE = 16_000  # Qwen2AudioEncoder native rate

    def __init__(
        self,
        model_name: str,
        revision: str,
        device: str | None = None,
        num_frames: int = 32,
        max_audio_length_seconds: float = 30.0,
        **kwargs: Any,
    ) -> None:
        self.device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self.num_frames = num_frames
        self.max_audio_samples = int(
            max_audio_length_seconds * self.AUDIO_SAMPLING_RATE
        )

        # VILA's vision tower and mm_projector are loaded as fp16 internally
        # and cannot be uniformly cast to bf16 (some submodules retain fp16
        # weights). Load LLM as fp16 to match for consistent multimodal forward.
        self.model = AutoModel.from_pretrained(
            model_name,
            revision=revision,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            **kwargs,
        )
        self.model.eval()
        self.model.to(self.device)

        self.processor = AutoProcessor.from_pretrained(
            model_name, revision=revision, trust_remote_code=True
        )

        self.model.config.num_video_frames = num_frames
        self.processor.config.num_video_frames = num_frames
        # Audio arrives as a separate column in MTEB, not embedded in video
        self.model.config.load_audio_in_video = False
        self.processor.config.load_audio_in_video = False

    @staticmethod
    def _save_frames_as_video(frames: torch.Tensor) -> str:
        """Write a ``(T, C, H, W)`` uint8 tensor to a temporary MP4 file."""
        import torchvision

        fd, path = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)
        # torchvision.io.write_video expects (T, H, W, C) uint8
        torchvision.io.write_video(
            path, frames.permute(0, 2, 3, 1).contiguous(), fps=24
        )
        return path

    @staticmethod
    def _save_image_as_file(image: Any) -> str:
        """Write a PIL Image to a temporary PNG file (omnivinci processor expects paths)."""
        fd, path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        # PNG only supports RGB/RGBA; convert CMYK/L/etc. before saving.
        if image.mode not in ("RGB", "RGBA", "L", "LA", "P"):
            image = image.convert("RGB")
        image.save(path)
        return path

    @staticmethod
    def _save_audio_as_wav(audio_data: Any, fallback_sr: int) -> str:
        """Write an audio array or ``AudioInputItem`` dict to a temporary WAV."""
        import numpy as np
        import soundfile as sf

        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)

        if isinstance(audio_data, dict) and "array" in audio_data:
            array = audio_data["array"]
            sr = audio_data.get("sampling_rate", fallback_sr)
        else:
            array = audio_data
            sr = fallback_sr

        if isinstance(array, torch.Tensor):
            array = array.numpy()
        if not isinstance(array, np.ndarray):
            array = np.asarray(array, dtype=np.float32)

        sf.write(path, array, sr)
        return path

    def _build_conversations(
        self, batch: BatchedInput
    ) -> tuple[list[list[dict]], list[str]]:
        """Build VILA conversations from a batch, returning conversations and temp file paths."""
        texts = batch.get("text", [])
        images = batch.get("image", [])
        audios = batch.get("audio", [])
        videos = batch.get("video", [])
        batch_size = max(len(texts), len(images), len(audios), len(videos))

        conversations = []
        temp_files: list[str] = []
        for i in range(batch_size):
            content: list[dict[str, Any]] = []

            if i < len(videos) and videos[i] is not None:
                path = self._save_frames_as_video(videos[i])
                temp_files.append(path)
                content.append({"type": "video", "video": path})

            if i < len(audios) and audios[i] is not None:
                path = self._save_audio_as_wav(audios[i], self.AUDIO_SAMPLING_RATE)
                temp_files.append(path)
                content.append({"type": "audio", "audio": path})

            if i < len(images) and images[i] is not None:
                path = self._save_image_as_file(images[i])
                temp_files.append(path)
                content.append({"type": "image", "image": path})

            content.append({"type": "text", "text": texts[i] if i < len(texts) else ""})
            conversations.append([{"role": "user", "content": content}])

        return conversations, temp_files

    def _encode_batch(self, batch: BatchedInput) -> torch.Tensor:
        """Encode a batch of multimodal samples and return L2-normalised embeddings."""
        conversations, temp_files = self._build_conversations(batch)
        try:
            text_prompts = [
                self.processor.apply_chat_template(
                    conv, tokenize=False, add_generation_prompt=False
                )
                for conv in conversations
            ]
            inputs = self.processor(text_prompts)

            outputs = self.model(
                input_ids=inputs.input_ids.to(self.device),
                media=getattr(inputs, "media", None),
                media_config=getattr(inputs, "media_config", None),
                output_hidden_states=True,
                return_dict=True,
            )

            embeddings = outputs.hidden_states[-1][:, -1]
            return torch.nn.functional.normalize(embeddings, p=2, dim=-1)

        finally:
            for f in temp_files:
                try:
                    pathlib.Path(f).unlink()
                except OSError:
                    pass

    @torch.inference_mode()
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
        has_video = "video" in inputs.dataset.features
        has_audio = "audio" in inputs.dataset.features

        if has_video or has_audio:
            inputs.collate_fn = VideoCollator(
                target_sampling_rate=self.AUDIO_SAMPLING_RATE,
                num_frames=self.num_frames,
                max_samples=self.max_audio_samples,
            )

        all_embeddings: list[torch.Tensor] = []
        for batch in tqdm(inputs, desc="Encoding"):
            embeddings = self._encode_batch(batch)
            all_embeddings.append(embeddings.cpu())

        return torch.cat(all_embeddings, dim=0).float()


_OMNIVINCI_CITATION = r"""
@article{ye2025omnivinci,
    title={OmniVinci: Enhancing Architecture and Data for Omni-Modal Understanding LLM},
    author={Ye, Hanrong and Yang, Chao-Han Huck and Goel, Arushi and Huang, Wei and Zhu, Ligeng and Su, Yuanhang and Lin, Sean and Cheng, An-Chieh and Wan, Zhen and Tian, Jinchuan and others},
    journal={arXiv preprint arXiv:2510.15870},
    year={2025}
}
"""

omnivinci = ModelMeta(
    loader=OmniVinciWrapper,
    name="nvidia/omnivinci",
    revision="7b3777b1c1f7c4e85f7bdef7b765dee1e76c1b7f",
    release_date="2025-10-21",
    languages=["eng-Latn"],
    n_parameters=8_742_888_688,
    memory_usage_mb=33351,
    max_tokens=32768,
    embed_dim=3584,
    n_embedding_parameters=543_553_024,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/NVlabs/OmniVinci",
    public_training_data=None,
    framework=["PyTorch", "Transformers", "safetensors"],
    reference="https://huggingface.co/nvidia/omnivinci",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=True,
    training_datasets=None,
    adapted_from="Qwen/Qwen2-7B",
    superseded_by=None,
    modalities=["text", "image", "audio", "video"],
    model_type=["dense"],
    citation=_OMNIVINCI_CITATION,
    extra_requirements_groups=["omnivinci"],
)
