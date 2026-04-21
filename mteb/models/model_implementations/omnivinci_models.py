from __future__ import annotations

import os
import pathlib
import tempfile
from typing import TYPE_CHECKING, Any

import torch
from tqdm.auto import tqdm

from mteb._create_dataloaders import VideoCollator
from mteb._requires_package import (
    requires_audio_dependencies,
    requires_image_dependencies,
)
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput, PromptType


class OmniVinciWrapper(AbsEncoder):
    """MTEB wrapper for NVIDIA OmniVinci.

    OmniVinci is an omni-modal understanding LLM built on the VILA architecture
    with a Qwen2 backbone, SiGLIP vision encoder, and Qwen2AudioEncoder.
    Supports text, image, audio, and video modalities.

    Since the VILA processor expects file paths for video and audio inputs,
    decoded frames and audio arrays from MTEB are written to temporary files
    before being passed to the processor. Images (PIL) are passed directly.

    Uses last-token pooling over the final hidden states for embeddings.
    """

    AUDIO_SAMPLING_RATE = 16_000  # Qwen2AudioEncoder native rate

    def __init__(
        self,
        model_name: str,
        revision: str,
        device: str | None = None,
        num_frames: int = 32,
        **kwargs: Any,
    ) -> None:
        requires_image_dependencies()
        requires_audio_dependencies()

        from transformers import AutoModel, AutoProcessor
        from transformers.modeling_utils import PreTrainedModel
        from transformers.utils.import_utils import is_flash_attn_2_available

        self.device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self.num_frames = num_frames

        attn_implementation = (
            "flash_attention_2" if is_flash_attn_2_available() else "sdpa"
        )
        # VILA's remote code hardcodes flash_attention_2 in several
        # sub-model from_pretrained calls (siglip_encoder, qwen_audio_encoder).
        # When flash-attn is not installed, temporarily intercept
        # from_pretrained to replace flash_attention_2 with our fallback.
        if attn_implementation != "flash_attention_2":
            _orig_from_pretrained = PreTrainedModel.from_pretrained.__func__

            @classmethod  # type: ignore[misc]
            def _patched_from_pretrained(cls, *args, **kw):
                if kw.get("attn_implementation") == "flash_attention_2":
                    kw["attn_implementation"] = attn_implementation
                return _orig_from_pretrained(cls, *args, **kw)

            PreTrainedModel.from_pretrained = _patched_from_pretrained

        self.model = AutoModel.from_pretrained(
            model_name,
            revision=revision,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation=attn_implementation,
            **kwargs,
        )

        # Restore original from_pretrained
        if attn_implementation != "flash_attention_2":
            PreTrainedModel.from_pretrained = _orig_from_pretrained
        self.model.eval()
        self.model.to(self.device)

        self.processor = AutoProcessor.from_pretrained(
            model_name, revision=revision, trust_remote_code=True
        )

        # Configure video/audio processing
        self.model.config.num_video_frames = num_frames
        self.processor.config.num_video_frames = num_frames
        # Audio arrives as a separate column in MTEB, not embedded in video
        self.model.config.load_audio_in_video = False
        self.processor.config.load_audio_in_video = False

        # MTEB's VideoCollator already decodes and samples frames.  Patch
        # VILA's _load_video so that when our wrapper passes in-memory
        # frames via the IN_MEMORY_VIDEO sentinel, it bypasses cv2 and
        # returns them directly — avoiding a lossy MP4 round-trip.
        self._video_registry: dict[str, list[Any]] = {}
        self._patch_load_video()

    # ------------------------------------------------------------------
    # In-memory video bridge (avoids lossy MP4 round-trip)
    # ------------------------------------------------------------------

    def _patch_load_video(self) -> None:
        """Patch VILA's ``_load_video`` to consume our in-memory frame lists.

        VILA's media pipeline expects a file path and re-decodes it via cv2.
        MTEB has already decoded and sampled frames for us, so we register
        the resulting PIL list under a marker file and intercept the load.
        """
        import sys

        cached = next(
            (
                m
                for name, m in sys.modules.items()
                if name.endswith(".media") and "omnivinci" in name
            ),
            None,
        )
        if cached is None or not hasattr(cached, "_load_video"):
            return

        registry = self._video_registry
        original = cached._load_video

        def _patched(video_path, *args, **kwargs):
            if isinstance(video_path, str) and video_path in registry:
                frames = registry[video_path]
                # MTEB has already decoded + sampled frames.  We don't know
                # the source FPS, so use evenly-spaced 1s timestamps; the
                # media encoder only needs a monotonic sequence for
                # temporal position embedding.
                video_info = {
                    "video_path": video_path,
                    "has_audio": False,
                    "video_duration": float(len(frames)),
                    "audio_info": None,
                    "video_frame_times": [float(i) for i in range(len(frames))],
                }
                return frames, None, video_info
            return original(video_path, *args, **kwargs)

        cached._load_video = _patched

    def _frames_to_pil(self, frames: torch.Tensor) -> list[Any]:
        """Convert a ``(T, C, H, W)`` uint8 tensor to a list of PIL images."""
        import PIL.Image

        return [
            PIL.Image.fromarray(
                frame.permute(1, 2, 0).cpu().numpy().astype("uint8")
            )
            for frame in frames
        ]

    @staticmethod
    def _save_audio_as_wav(audio_data: Any, fallback_sr: int) -> str:
        """Write an audio array to a temporary WAV file.

        Args:
            audio_data: Either a raw numpy/torch array or an
                :class:`~mteb.types._encoder_io.AudioInputItem` dict
                with ``array`` and ``sampling_rate`` keys.
            fallback_sr: Sampling rate to use when *audio_data* is a raw array.

        Returns:
            Path to the temporary WAV file (caller must unlink).
        """
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

    # ------------------------------------------------------------------
    # Core encoding
    # ------------------------------------------------------------------

    def _encode_single(
        self,
        text: str = "",
        image: Any = None,
        audio: Any = None,
        video: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode one multimodal sample and return its L2-normalised embedding."""
        temp_files: list[str] = []
        video_key: str | None = None
        try:
            content: list[dict[str, Any]] = []

            if video is not None:
                # Create an empty marker file so VILA's processor path
                # validation (osp.exists) passes; register PIL frames under
                # the same path so the patched _load_video returns them
                # without re-decoding.
                fd, video_key = tempfile.mkstemp(suffix=".mp4")
                os.close(fd)
                temp_files.append(video_key)
                self._video_registry[video_key] = self._frames_to_pil(video)
                content.append({"type": "video", "video": video_key})

            if audio is not None:
                audio_path = self._save_audio_as_wav(audio, self.AUDIO_SAMPLING_RATE)
                temp_files.append(audio_path)
                content.append({"type": "audio", "audio": audio_path})

            if image is not None:
                # VILA processor accepts PIL Images directly
                content.append({"type": "image", "image": image})

            content.append({"type": "text", "text": text})

            conversation = [{"role": "user", "content": content}]
            text_prompt = self.processor.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=False
            )

            inputs = self.processor([text_prompt])
            input_ids = inputs.input_ids.to(self.device)

            outputs = self.model(
                input_ids=input_ids,
                media=getattr(inputs, "media", None),
                media_config=getattr(inputs, "media_config", None),
                output_hidden_states=True,
                return_dict=True,
            )

            # Last-token pooling
            hidden_states = outputs.hidden_states[-1]
            embedding = hidden_states[:, -1]
            return torch.nn.functional.normalize(embedding, p=2, dim=-1)

        finally:
            for f in temp_files:
                try:
                    pathlib.Path(f).unlink()
                except OSError:
                    pass
            if video_key is not None:
                self._video_registry.pop(video_key, None)

    @torch.no_grad()
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
                max_frames=self.num_frames,
            )

        all_embeddings: list[torch.Tensor] = []
        for batch in tqdm(inputs, desc="Encoding"):
            batch_texts = batch.get("text", [])
            batch_images = batch.get("image", [])
            batch_audio = batch.get("audio", [])
            batch_video = batch.get("video", [])

            batch_size = max(
                len(batch_texts),
                len(batch_images),
                len(batch_audio),
                len(batch_video),
            )

            for i in range(batch_size):
                text = batch_texts[i] if i < len(batch_texts) else ""
                image = batch_images[i] if i < len(batch_images) else None
                audio = batch_audio[i] if i < len(batch_audio) else None
                video = batch_video[i] if i < len(batch_video) else None

                emb = self._encode_single(text, image, audio, video)
                all_embeddings.append(emb.cpu())

        return torch.cat(all_embeddings, dim=0).float()


# ---------------------------------------------------------------------------
# Model metadata
# ---------------------------------------------------------------------------

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
)
