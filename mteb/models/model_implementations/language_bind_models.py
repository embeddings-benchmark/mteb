from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from tqdm.auto import tqdm

from mteb.models import ModelMeta
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.modality_collators import AudioCollator, VideoCollator


def _patch_compatibility() -> None:
    """Compatibility shims so the unmaintained LanguageBind library imports
    cleanly against current ``transformers`` / ``torchvision`` / ``torchaudio``.

    Each shim is independent and fail-soft: a single broken patch only emits a
    warning so the rest of the module still imports.
    """
    try:
        import transformers.models.clip.modeling_clip as _clip_mod

        if not hasattr(_clip_mod, "_expand_mask"):

            def _expand_mask(
                mask: torch.Tensor,
                dtype: torch.dtype,
                tgt_len: int | None = None,
            ) -> torch.Tensor:
                bsz, src_len = mask.size()
                tgt_len = tgt_len if tgt_len is not None else src_len
                expanded_mask = (
                    mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
                )
                inverted_mask = 1.0 - expanded_mask
                return inverted_mask.masked_fill(inverted_mask.to(bool), float("-inf"))

            _clip_mod._expand_mask = _expand_mask  # type: ignore[attr-defined]
    except Exception as e:
        warnings.warn(f"LanguageBind compat patch `_expand_mask` failed: {e}")

    try:
        import transformers.models.clip.modeling_clip as _clip_mod

        if not hasattr(_clip_mod, "clip_loss"):
            import torch.nn as _nn

            def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
                caption_loss = _nn.functional.cross_entropy(
                    similarity,
                    torch.arange(len(similarity), device=similarity.device),
                )
                image_loss = _nn.functional.cross_entropy(
                    similarity.t(),
                    torch.arange(len(similarity), device=similarity.device),
                )
                return (caption_loss + image_loss) / 2.0

            _clip_mod.clip_loss = clip_loss  # type: ignore[attr-defined]
    except Exception as e:
        warnings.warn(f"LanguageBind compat patch `clip_loss` failed: {e}")

    try:
        import sys

        try:
            import torchvision.transforms.functional_tensor  # type: ignore[import-not-found] # noqa: F401
        except ImportError:
            import torchvision.transforms.functional as _tv_functional

            sys.modules["torchvision.transforms.functional_tensor"] = _tv_functional
    except Exception as e:
        warnings.warn(
            f"LanguageBind compat patch `torchvision.transforms.functional_tensor` failed: {e}"
        )

    try:
        import torchaudio  # type: ignore[import-not-found]

        if not hasattr(torchaudio, "set_audio_backend"):
            torchaudio.set_audio_backend = lambda backend: None  # type: ignore[attr-defined]
    except Exception as e:
        warnings.warn(
            f"LanguageBind compat patch `torchaudio.set_audio_backend` failed: {e}"
        )


_patch_compatibility()


def _patch_languagebind_tokenizer(tokenizer_cls: type) -> None:
    """Force a LanguageBind tokenizer subclass to use ``CLIPTokenizer.__init__``.

    LanguageBind's tokenizer subclasses inherit from ``CLIPTokenizer`` but
    override ``__init__`` with a positional signature that breaks under
    newer ``transformers`` releases. We re-route through
    ``CLIPTokenizer.__init__`` with explicit keyword arguments.
    """
    if getattr(tokenizer_cls, "_mteb_patched", False):
        return
    try:
        from transformers import CLIPTokenizer

        def _patched_init(
            self: Any,
            vocab_file: str,
            merges_file: str,
            errors: str = "replace",
            unk_token: str = "<|endoftext|>",  # noqa: S107
            bos_token: str = "<|startoftext|>",  # noqa: S107
            eos_token: str = "<|endoftext|>",  # noqa: S107
            pad_token: str = "<|endoftext|>",  # noqa: S107
            **kwargs: Any,
        ) -> None:
            CLIPTokenizer.__init__(
                self,
                vocab_file=vocab_file,
                merges_file=merges_file,
                errors=errors,
                unk_token=unk_token,
                bos_token=bos_token,
                eos_token=eos_token,
                pad_token=pad_token,
                **kwargs,
            )

        tokenizer_cls.__init__ = _patched_init  # type: ignore[method-assign]
        tokenizer_cls._mteb_patched = True  # type: ignore[attr-defined]
    except Exception as e:
        warnings.warn(
            f"LanguageBind tokenizer patch for {tokenizer_cls.__name__} failed: {e}"
        )


if TYPE_CHECKING:
    from torch.utils.data import DataLoader
    from typing_extensions import Unpack

    from mteb import TaskMetadata
    from mteb.types import Array, BatchedInput, EncodeKwargs, PromptType
    from mteb.types._encoder_io import AudioInput, ImageInput, TextInput, VideoInput


# LanguageBind expects audio sampled at 16 kHz (its audio mel-spectrogram pipeline).
_LANGUAGE_BIND_AUDIO_SR = 16000

_LANGUAGE_BIND_SETUP_DOC = """
    Setup::

        git clone https://github.com/PKU-YuanGroup/LanguageBind.git
        export PYTHONPATH="/path/to/LanguageBind:$PYTHONPATH"
        pip install einops decord opencv-python-headless pytorchvideo peft
"""


class _LanguageBindBase(AbsEncoder):
    """Shared text-encoding logic for the LanguageBind modality wrappers.

    Subclasses load their own modality-specific checkpoint into ``self.model``
    (with a paired ``self.tokenizer``) and implement the modality-specific
    ``get_*_embeddings`` plus ``encode`` methods. The text tower is the same
    OpenCLIP encoder across all LanguageBind variants, so the tokenization
    and text-projection path lives here.
    """

    model: Any
    tokenizer: Any
    device: str

    def _tokenize(self, texts: list[str]) -> dict[str, torch.Tensor]:
        # max_length=77 is the OpenCLIP / CLIP tokenizer default context length
        # that LanguageBind inherits from its OpenCLIP base.
        tokens = self.tokenizer(
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

        for batch in tqdm(
            inputs,
            disable=not show_progress_bar,
            desc="Processing text batches",
        ):
            texts = list(batch["text"])
            tokens = self._tokenize(texts)

            with torch.autocast(str(self.device), dtype=torch.bfloat16):
                text_outputs = self.model.text_model(
                    input_ids=tokens["input_ids"],
                    attention_mask=tokens.get("attention_mask"),
                )
                text_embeds = self.model.text_projection(text_outputs[1])
                text_embeds /= text_embeds.norm(dim=-1, keepdim=True)
                all_embeddings.append(text_embeds.cpu().float().numpy())

        return np.vstack(all_embeddings)


class LanguageBindVideoWrapper(_LanguageBindBase):
    """MTEB wrapper for LanguageBind video + text.

    Video frames arrive pre-decoded via the VideoCollator. The public
    LanguageBind processor expects file paths, so we apply the processor's
    transform directly to the frame tensor and skip the file-loading step.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        fps: float | None = None,
        max_frames: int | None = None,
        num_frames: int | None = 8,
        max_samples: int | None = None,
        **kwargs: Any,
    ):
        from languagebind import (
            LanguageBindVideo,
            LanguageBindVideoProcessor,
            LanguageBindVideoTokenizer,
        )

        _patch_languagebind_tokenizer(LanguageBindVideoTokenizer)

        self.model_name = model_name
        self.device = device
        self.fps = fps
        self.max_frames = max_frames
        self.num_frames = num_frames
        self.max_samples = max_samples
        self.sampling_rate = _LANGUAGE_BIND_AUDIO_SR

        self.model = LanguageBindVideo.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.tokenizer = LanguageBindVideoTokenizer.from_pretrained(model_name)
        self.processor = LanguageBindVideoProcessor(self.model.config, self.tokenizer)

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
        transformed = self.processor.transform({"video": video})
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
        inputs.collate_fn = VideoCollator(
            target_sampling_rate=self.sampling_rate,
            fps=self.fps,
            max_frames=self.max_frames,
            num_frames=self.num_frames,
            max_samples=self.max_samples,
        )

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
                vision_outputs = self.model.vision_model(pixel_values=processed)
                video_embeds = self.model.visual_projection(vision_outputs[1])
                video_embeds /= video_embeds.norm(dim=-1, keepdim=True)
                all_embeddings.append(video_embeds.cpu().float().numpy())

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
        features = inputs.dataset.features
        has_text = "text" in features
        has_video = "video" in features

        embeddings = None
        if has_text:
            text_emb = self.get_text_embeddings(
                inputs, prompt_type=prompt_type, **kwargs
            )
            embeddings = text_emb if embeddings is None else embeddings + text_emb
        if has_video:
            video_emb = self.get_video_embeddings(inputs, **kwargs)
            embeddings = video_emb if embeddings is None else embeddings + video_emb

        if embeddings is not None:
            return embeddings

        raise ValueError(
            f"No supported modality found in dataset features: {list(features.keys())}"
        )


class LanguageBindAudioWrapper(_LanguageBindBase):
    """MTEB wrapper for LanguageBind audio + text."""

    def __init__(
        self,
        model_name: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs: Any,
    ):
        from languagebind import (
            LanguageBindAudio,
            LanguageBindAudioProcessor,
            LanguageBindAudioTokenizer,
        )

        _patch_languagebind_tokenizer(LanguageBindAudioTokenizer)

        self.model_name = model_name
        self.device = device
        self.sampling_rate = _LANGUAGE_BIND_AUDIO_SR

        self.model = LanguageBindAudio.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.tokenizer = LanguageBindAudioTokenizer.from_pretrained(model_name)
        self.processor = LanguageBindAudioProcessor(self.model.config, self.tokenizer)

    def _transform_audio(self, audio_array: np.ndarray) -> torch.Tensor:
        """Apply LanguageBind's audio transform to a raw waveform array."""
        waveform = torch.as_tensor(audio_array, dtype=torch.float32)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        return self.processor.transform((waveform, self.sampling_rate))

    @torch.inference_mode()
    def get_audio_embeddings(
        self,
        inputs: DataLoader[AudioInput],
        show_progress_bar: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        """Get audio-only embeddings."""
        all_embeddings = []
        inputs.collate_fn = AudioCollator(target_sampling_rate=self.sampling_rate)

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
                audio_outputs = self.model.vision_model(pixel_values=processed)
                audio_embeds = self.model.visual_projection(audio_outputs[1])
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
        features = inputs.dataset.features
        has_text = "text" in features
        has_audio = "audio" in features

        embeddings = None
        if has_text:
            text_emb = self.get_text_embeddings(
                inputs, prompt_type=prompt_type, **kwargs
            )
            embeddings = text_emb if embeddings is None else embeddings + text_emb
        if has_audio:
            audio_emb = self.get_audio_embeddings(inputs, **kwargs)
            embeddings = audio_emb if embeddings is None else embeddings + audio_emb

        if embeddings is not None:
            return embeddings

        raise ValueError(
            f"No supported modality found in dataset features: {list(features.keys())}"
        )


class LanguageBindImageWrapper(_LanguageBindBase):
    """MTEB wrapper for LanguageBind image + text."""

    def __init__(
        self,
        model_name: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs: Any,
    ):
        from languagebind import (
            LanguageBindImage,
            LanguageBindImageProcessor,
            LanguageBindImageTokenizer,
        )

        _patch_languagebind_tokenizer(LanguageBindImageTokenizer)

        self.model_name = model_name
        self.device = device

        self.model = LanguageBindImage.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.tokenizer = LanguageBindImageTokenizer.from_pretrained(model_name)
        self.processor = LanguageBindImageProcessor(self.model.config, self.tokenizer)

    @torch.inference_mode()
    def get_image_embeddings(
        self,
        inputs: DataLoader[ImageInput],
        show_progress_bar: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        """Get image-only embeddings."""
        all_embeddings = []

        for batch in tqdm(
            inputs,
            disable=not show_progress_bar,
            desc="Processing image batches",
        ):
            images = list(batch["image"])
            processed = torch.stack(
                [self.processor.transform(img) for img in images]
            ).to(self.device)

            with torch.autocast(str(self.device), dtype=torch.bfloat16):
                vision_outputs = self.model.vision_model(pixel_values=processed)
                image_embeds = self.model.visual_projection(vision_outputs[1])
                image_embeds /= image_embeds.norm(dim=-1, keepdim=True)
                all_embeddings.append(image_embeds.cpu().float().numpy())

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
        features = inputs.dataset.features
        has_text = "text" in features
        has_image = "image" in features

        embeddings = None
        if has_text:
            text_emb = self.get_text_embeddings(
                inputs, prompt_type=prompt_type, **kwargs
            )
            embeddings = text_emb if embeddings is None else embeddings + text_emb
        if has_image:
            image_emb = self.get_image_embeddings(inputs, **kwargs)
            embeddings = image_emb if embeddings is None else embeddings + image_emb

        if embeddings is not None:
            return embeddings

        raise ValueError(
            f"No supported modality found in dataset features: {list(features.keys())}"
        )


class LanguageBindOmniWrapper(AbsEncoder):
    """MTEB wrapper for LanguageBind video + text.

    Video frames arrive pre-decoded via the VideoCollator. The public
    LanguageBind processor expects file paths, so we apply the processor's
    transform directly to the frame tensor and skip the file-loading step.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        fps: float | None = None,
        max_frames: int | None = None,
        num_frames: int | None = 8,
        max_samples: int | None = None,
        **kwargs: Any,
    ):
        self.video_model = LanguageBindVideoWrapper(
            "LanguageBind/LanguageBind_Video_FT",
            device=device,
            fps=fps,
            max_frames=max_frames,
            num_frames=num_frames,
            max_samples=max_samples,
        )
        self.audio_model = LanguageBindAudioWrapper(
            "LanguageBind/LanguageBind_Audio_FT",
            device=device,
        )
        self.image_model = LanguageBindImageWrapper(
            "LanguageBind/LanguageBind_Image",
            device=device,
        )

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
        features = inputs.dataset.features
        has_text = "text" in features
        has_audio = "audio" in features
        has_video = "video" in features
        has_image = "image" in features

        embeddings = None
        if has_text:
            text_emb = self.video_model.get_text_embeddings(
                inputs, prompt_type=prompt_type, **kwargs
            )
            embeddings = text_emb if embeddings is None else embeddings + text_emb
        if has_image:
            image_emb = self.image_model.get_image_embeddings(inputs, **kwargs)
            embeddings = image_emb if embeddings is None else embeddings + image_emb
        if has_audio:
            audio_emb = self.audio_model.get_audio_embeddings(inputs, **kwargs)
            embeddings = audio_emb if embeddings is None else embeddings + audio_emb
        if has_video:
            video_emb = self.video_model.get_video_embeddings(inputs, **kwargs)
            embeddings = video_emb if embeddings is None else embeddings + video_emb
        return embeddings


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
    max_tokens=77,
    embed_dim=768,
)


language_bind_video_ft = ModelMeta(
    loader=LanguageBindVideoWrapper,
    name="LanguageBind/LanguageBind_Video_FT",
    revision="13f52c20ce666a7d017bcd00522039f4ab034a66",
    n_parameters=427_616_513,
    n_embedding_parameters=37_945_344,
    memory_usage_mb=1631,
    modalities=["video", "text"],
    loader_kwargs=dict(num_frames=8),
    **_LANGUAGE_BIND_COMMON,
)

language_bind_audio_ft = ModelMeta(
    loader=LanguageBindAudioWrapper,
    name="LanguageBind/LanguageBind_Audio_FT",
    revision="4820c496563c46acfb1ff9a486fae5319f16257e",
    n_parameters=345_000_000,
    n_embedding_parameters=37_945_344,
    memory_usage_mb=1316,
    modalities=["audio", "text"],
    loader_kwargs=dict(),
    **_LANGUAGE_BIND_COMMON,
)

language_bind_image = ModelMeta(
    loader=LanguageBindImageWrapper,
    name="LanguageBind/LanguageBind_Image",
    revision="d8c2e37b439f4fc47c649dc8b90cdcd3a4e0c80e",
    n_parameters=427_616_513,
    n_embedding_parameters=37_945_344,
    memory_usage_mb=1631,
    modalities=["image", "text"],
    loader_kwargs=dict(),
    **_LANGUAGE_BIND_COMMON,
)

language_bind_omni = ModelMeta(
    loader=LanguageBindOmniWrapper,
    name="LanguageBind/LanguageBind_Omni",
    revision="d8c2e37b439f4fc47c649dc8b90cdcd3a4e0c80e",
    n_parameters=427_616_513 + 345_000_000 + 427_616_513,
    n_embedding_parameters=37_945_344,
    memory_usage_mb=1631 + 1316 + 1631,
    modalities=["image", "text", "audio", "video"],
    loader_kwargs=dict(num_frames=8),
    **_LANGUAGE_BIND_COMMON,
)
