from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from tqdm.auto import tqdm

from mteb.models.modality_collators import VideoCollator
from mteb._requires_package import requires_package
from mteb.models import ModelMeta
from mteb.models.abs_encoder import AbsEncoder

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from mteb import TaskMetadata
    from mteb.types import Array, BatchedInput, PromptType
    from mteb.types._encoder_io import AudioInput, TextInput, VideoInput


# LanguageBind expects audio sampled at 16 kHz (its audio mel-spectrogram pipeline).
_LANGUAGE_BIND_AUDIO_SR = 16000

_VIDEO_MODEL_NAME = "LanguageBind/LanguageBind_Video_FT"
_AUDIO_MODEL_NAME = "LanguageBind/LanguageBind_Audio_FT"


class LanguageBindWrapper(AbsEncoder):
    """MTEB wrapper for LanguageBind (video / audio / text).

    LanguageBind aligns audio, video and text into a shared OpenCLIP-style
    embedding space. The library exposes separate checkpoints for each
    non-text modality but they share the same text encoder, so we load the
    video and audio variants together and route encode calls per modality.

    Video frames arrive pre-decoded via the VideoCollator. The public
    LanguageBind processor expects file paths, so for video we apply the
    processor's transform directly to the frame tensor and skip the
    file-loading step.
    """

    def __init__(
        self,
        model_name: str = _VIDEO_MODEL_NAME,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        fps: float | None = None,
        max_frames: int | None = None,
        num_frames: int | None = 8,
        max_samples: int | None = None,
        **kwargs: Any,
    ):
        requires_package(
            self,
            "languagebind",
            model_name,
            install_instruction="pip install git+https://github.com/PKU-YuanGroup/LanguageBind.git",
        )
        from languagebind import (
            LanguageBindAudio,
            LanguageBindAudioProcessor,
            LanguageBindAudioTokenizer,
            LanguageBindVideo,
            LanguageBindVideoProcessor,
            LanguageBindVideoTokenizer,
        )

        self.model_name = model_name
        self.device = device
        self.fps = fps
        self.max_frames = max_frames
        self.num_frames = num_frames
        self.max_samples = max_samples
        self.sampling_rate = _LANGUAGE_BIND_AUDIO_SR

        self.video_model = LanguageBindVideo.from_pretrained(_VIDEO_MODEL_NAME).to(
            self.device
        )
        self.video_model.eval()
        self.video_tokenizer = LanguageBindVideoTokenizer.from_pretrained(
            _VIDEO_MODEL_NAME
        )
        self.video_processor = LanguageBindVideoProcessor(
            self.video_model.config, self.video_tokenizer
        )

        self.audio_model = LanguageBindAudio.from_pretrained(_AUDIO_MODEL_NAME).to(
            self.device
        )
        self.audio_model.eval()
        self.audio_tokenizer = LanguageBindAudioTokenizer.from_pretrained(
            _AUDIO_MODEL_NAME
        )
        self.audio_processor = LanguageBindAudioProcessor(
            self.audio_model.config, self.audio_tokenizer
        )

    def _text_model(self) -> Any:
        # Audio and video checkpoints share the same text-tower architecture;
        # use the video-aligned tower to match the joint space we expose.
        return self.video_model

    def _tokenize(self, texts: list[str]) -> dict[str, torch.Tensor]:
        tokens = self.video_tokenizer(
            texts,
            max_length=77,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {k: v.to(self.device) for k, v in tokens.items()}

    @torch.inference_mode()
    def get_text_embeddings(
        self,
        inputs: DataLoader[TextInput],
        prompt_type: PromptType | None = None,
        show_progress_bar: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        """Get text embeddings aligned to the LanguageBind joint space."""
        all_embeddings = []
        text_model = self._text_model()

        for batch in tqdm(
            inputs,
            disable=not show_progress_bar,
            desc="Processing text batches",
        ):
            texts = list(batch["text"])
            tokens = self._tokenize(texts)

            with torch.autocast(str(self.device), dtype=torch.bfloat16):
                text_outputs = text_model.text_model(
                    input_ids=tokens["input_ids"],
                    attention_mask=tokens.get("attention_mask"),
                )
                text_embeds = text_model.text_projection(text_outputs[1])
                text_embeds /= text_embeds.norm(dim=-1, keepdim=True)
                all_embeddings.append(text_embeds.cpu().float().numpy())

        return np.vstack(all_embeddings)

    def _transform_video_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """Apply LanguageBind's video transform to pre-decoded frames.

        torchcodec yields (T, H, W, C) uint8 frames; LanguageBind expects
        (C, T, H, W) float ready for the OpenCLIP visual tower.
        """
        if frames.ndim != 4:
            raise ValueError(
                f"Expected 4D video tensor (T, H, W, C); got shape {tuple(frames.shape)}"
            )
        # (T, H, W, C) -> (C, T, H, W)
        video = frames.permute(3, 0, 1, 2).float()
        transformed = self.video_processor.transform({"video": video})
        return transformed["video"]

    @torch.inference_mode()
    def get_video_embeddings(
        self,
        inputs: DataLoader[VideoInput],
        show_progress_bar: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        """Get video-only embeddings."""
        all_embeddings = []

        for batch in tqdm(
            inputs,
            disable=not show_progress_bar,
            desc="Processing video batches",
        ):
            frames_list = list(batch["video"])
            processed = torch.stack(
                [self._transform_video_frames(frames) for frames in frames_list]
            ).to(self.device)

            with torch.autocast(str(self.device), dtype=torch.bfloat16):
                vision_outputs = self.video_model.vision_model(pixel_values=processed)
                video_embeds = self.video_model.visual_projection(vision_outputs[1])
                video_embeds /= video_embeds.norm(dim=-1, keepdim=True)
                all_embeddings.append(video_embeds.cpu().float().numpy())

        return np.vstack(all_embeddings)

    def _transform_audio(self, audio_array: np.ndarray) -> torch.Tensor:
        """Apply LanguageBind's audio transform to a raw waveform array."""
        waveform = torch.as_tensor(audio_array, dtype=torch.float32)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        return self.audio_processor.transform(waveform)

    @torch.inference_mode()
    def get_audio_embeddings(
        self,
        inputs: DataLoader[AudioInput],
        show_progress_bar: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        """Get audio-only embeddings."""
        all_embeddings = []

        for batch in tqdm(
            inputs,
            disable=not show_progress_bar,
            desc="Processing audio batches",
        ):
            audio_arrays = [audio["array"] for audio in batch["audio"]]
            processed = torch.stack(
                [self._transform_audio(a) for a in audio_arrays]
            ).to(self.device)

            with torch.autocast(str(self.device), dtype=torch.bfloat16):
                audio_outputs = self.audio_model.vision_model(pixel_values=processed)
                audio_embeds = self.audio_model.visual_projection(audio_outputs[1])
                audio_embeds /= audio_embeds.norm(dim=-1, keepdim=True)
                all_embeddings.append(audio_embeds.cpu().float().numpy())

        return np.vstack(all_embeddings)

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
        has_audio = "audio" in inputs.dataset.features

        inputs.collate_fn = VideoCollator(
            target_sampling_rate=self.sampling_rate,
            fps=self.fps,
            max_frames=self.max_frames,
            num_frames=self.num_frames,
            max_samples=self.max_samples,
        )

        # Video-only
        if has_video and not has_audio and not has_text:
            return self.get_video_embeddings(inputs, **kwargs)

        # Audio-only
        if has_audio and not has_video and not has_text:
            return self.get_audio_embeddings(inputs, **kwargs)

        # Text-only
        if has_text and not has_video and not has_audio:
            return self.get_text_embeddings(inputs, prompt_type=prompt_type, **kwargs)

        # Mixed modality: fuse embeddings by addition
        embeddings = None

        if has_text:
            text_emb = self.get_text_embeddings(
                inputs, prompt_type=prompt_type, **kwargs
            )
            embeddings = text_emb

        if has_video:
            video_emb = self.get_video_embeddings(inputs, **kwargs)
            embeddings = video_emb if embeddings is None else embeddings + video_emb

        if has_audio:
            audio_emb = self.get_audio_embeddings(inputs, **kwargs)
            embeddings = audio_emb if embeddings is None else embeddings + audio_emb

        if embeddings is not None:
            return embeddings

        raise ValueError(
            f"No supported modality found in dataset features: {list(inputs.dataset.features.keys())}"
        )


# --- Model Metadata ---

_LANGUAGE_BIND_CITATION = r"""
@inproceedings{zhu2024languagebind,
      title={Language{B}ind: Extending Video-Language Pretraining to N-modality by Language-based Semantic Alignment},
      author={Bin Zhu and Bin Lin and Munan Ning and Yang Yan and Jiaxi Cui and Wang HongFa and Yatian Pang and Wenhao Jiang and Junwu Zhang and Zongwei Li and Cai Wan Zhang and Zhifeng Li and Wei Liu and Li Yuan},
      booktitle={The Twelfth International Conference on Learning Representations},
      year={2024},
      url={https://openreview.net/forum?id=QmZKc7UZCy},
}
"""

_LANGUAGE_BIND_COMMON = dict(
    languages=["eng-Latn"],
    release_date="2023-10-03",
    open_weights=True,
    public_training_code="https://github.com/PKU-YuanGroup/LanguageBind",
    public_training_data=None,
    framework=["PyTorch"],
    similarity_fn_name="cosine",
    use_instructions=False,
    training_datasets=set(),
    citation=_LANGUAGE_BIND_CITATION,
    license="mit",
    reference="https://github.com/PKU-YuanGroup/LanguageBind",
    extra_requirements_groups=[],
)


language_bind_video_ft = ModelMeta(
    loader=LanguageBindWrapper,
    name=_VIDEO_MODEL_NAME,
    revision="main",
    n_parameters=None,
    memory_usage_mb=None,
    max_tokens=77,
    embed_dim=768,
    modalities=["video", "text"],
    loader_kwargs=dict(num_frames=8),
    **_LANGUAGE_BIND_COMMON,
)

language_bind_audio_ft = ModelMeta(
    loader=LanguageBindWrapper,
    name=_AUDIO_MODEL_NAME,
    revision="main",
    n_parameters=None,
    memory_usage_mb=None,
    max_tokens=77,
    embed_dim=768,
    modalities=["audio", "text"],
    loader_kwargs=dict(),
    **_LANGUAGE_BIND_COMMON,
)
