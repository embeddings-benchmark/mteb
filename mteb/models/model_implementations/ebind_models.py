from __future__ import annotations

import tempfile
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from tqdm.auto import tqdm

from mteb._create_dataloaders import AudioCollator, VideoCollator
from mteb._requires_package import requires_package
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from mteb import TaskMetadata
    from mteb.types import Array, BatchedInput, PromptType
    from mteb.types._encoder_io import AudioInput, ImageInput, TextInput, VideoInput


class EBindWrapper(AbsEncoder):
    """MTEB wrapper for EBind multi-modal embedding model.

    EBind projects image, video, audio, and text into a shared 1024-dim
    embedding space using separate backbone encoders (Perception Encoder,
    ImageBind, Uni3D) with learned projection heads.

    EBind's processors expect file paths, not decoded data.  This wrapper
    bridges the gap:
    - **text**: passed directly (PETextProcessor accepts strings).
    - **image**: PIL images are fed to the image transform directly,
      bypassing the file-loading step in PEImageProcessor.
    - **video**: pre-decoded frame tensors are resized and normalised
      directly (matching ``ImageTransform._transform_torch_tensor``).
    - **audio**: numpy arrays are written to temporary ``.wav`` files
      because ImageBind's mel-spectrogram pipeline is non-trivial to
      replicate outside of ``torchaudio.load()``.
    """

    def __init__(
        self,
        model_name: str,
        revision: str | None = None,
        device: str | None = None,
        num_frames: int = 8,
        **kwargs: Any,
    ) -> None:
        requires_package(
            self,
            "ebind",
            model_name,
            install_instruction="pip install 'mteb[ebind]'",
        )
        requires_package(
            self,
            "soundfile",
            model_name,
            install_instruction="pip install 'mteb[ebind]'",
        )

        from ebind import EBindModel, EBindProcessor

        self.device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self.num_frames = num_frames

        self.model = EBindModel.from_pretrained(model_name, revision=revision)
        self.model.to(self.device).eval()

        self.processor = EBindProcessor.from_pretrained(
            model_name, revision=revision
        )
        self.processor = self.processor.to(self.device)

        # Grab the internal image transform so we can bypass file I/O
        # for images (the Compose pipeline accepts PIL Images directly).
        self._image_transform = self.processor.processors["image"].transform
        # PE image size (336 for PE-Core-L14-336) used for video frame resizing.
        self._image_size = self.processor.processors["image"].model_image_size

    # ------------------------------------------------------------------
    # Per-modality encode helpers
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def get_text_embeddings(
        self,
        inputs: DataLoader[TextInput],
        show_progress_bar: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        all_embeddings: list[np.ndarray] = []
        for batch in tqdm(inputs, disable=not show_progress_bar, desc="Encoding text"):
            # PETextProcessor accepts raw strings — use the processor directly.
            processed = self.processor({"text": batch["text"]}, return_tensors="pt")
            outputs = self.model.forward(**processed)
            all_embeddings.append(outputs["text"].cpu().float().numpy())
        return np.vstack(all_embeddings)

    @torch.inference_mode()
    def get_image_embeddings(
        self,
        inputs: DataLoader[ImageInput],
        show_progress_bar: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        all_embeddings: list[np.ndarray] = []
        for batch in tqdm(
            inputs, disable=not show_progress_bar, desc="Encoding images"
        ):
            # MTEB provides PIL Images.  PEImageProcessor opens file paths then
            # calls self.transform(pil_image), so we call the transform directly.
            img_tensors = torch.stack(
                [self._image_transform(img) for img in batch["image"]]
            ).to(self.device)
            outputs = self.model.forward(image=img_tensors)
            all_embeddings.append(outputs["image"].cpu().float().numpy())
        return np.vstack(all_embeddings)

    @torch.inference_mode()
    def get_audio_embeddings(
        self,
        inputs: DataLoader[AudioInput],
        show_progress_bar: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        import soundfile as sf

        all_embeddings: list[np.ndarray] = []
        for batch in tqdm(
            inputs, disable=not show_progress_bar, desc="Encoding audio"
        ):
            # IBAudioProcessor only accepts file paths (it calls
            # torchaudio.load internally for mel-spectrogram conversion).
            # Write decoded arrays to temporary .wav files.
            audio_tensors = []
            for item in batch["audio"]:
                with tempfile.NamedTemporaryFile(
                    suffix=".wav", delete=True
                ) as tmp:
                    sf.write(tmp.name, item["array"], item["sampling_rate"])
                    audio_tensors.append(
                        self.processor.processors["audio"](tmp.name)
                    )
            stacked = torch.stack(audio_tensors).to(self.device)
            outputs = self.model.forward(audio=stacked)
            all_embeddings.append(outputs["audio"].cpu().float().numpy())
        return np.vstack(all_embeddings)

    @torch.inference_mode()
    def get_video_embeddings(
        self,
        inputs: DataLoader[VideoInput],
        show_progress_bar: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        from torchvision.transforms.functional import InterpolationMode, normalize, resize

        all_embeddings: list[np.ndarray] = []
        for batch in tqdm(
            inputs, disable=not show_progress_bar, desc="Encoding video"
        ):
            # VideoCollator yields tensors of shape (T, C, H, W) uint8.
            # Apply the same preprocessing that PE's
            # VideoTransform._transform_torch_tensor uses: resize to
            # 336x336 BICUBIC, float [0,1], normalize mean=std=0.5.
            video_tensors = []
            for frames in batch["video"]:  # each is (T, C, H, W) uint8
                frames = resize(
                    frames,
                    [self._image_size, self._image_size],
                    interpolation=InterpolationMode.BICUBIC,
                    antialias=True,
                )
                frames = frames.float() / 255.0
                frames = normalize(frames, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                video_tensors.append(frames)
            stacked = torch.stack(video_tensors).to(self.device)
            outputs = self.model.forward(video=stacked)
            all_embeddings.append(outputs["video"].cpu().float().numpy())
        return np.vstack(all_embeddings)

    # ------------------------------------------------------------------
    # Main encode entry point
    # ------------------------------------------------------------------

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
        has_video = "video" in features
        has_audio = "audio" in features

        if has_video:
            inputs.collate_fn = VideoCollator(
                target_sampling_rate=16_000,
                max_frames=self.num_frames,
            )
        elif has_audio:
            inputs.collate_fn = AudioCollator(
                target_sampling_rate=16_000,
            )

        # Single-modality fast paths
        if has_text and not has_image and not has_video and not has_audio:
            return self.get_text_embeddings(inputs, **kwargs)

        if has_image and not has_text and not has_video and not has_audio:
            return self.get_image_embeddings(inputs, **kwargs)

        if has_video and not has_text and not has_image and not has_audio:
            return self.get_video_embeddings(inputs, **kwargs)

        if has_audio and not has_text and not has_image and not has_video:
            return self.get_audio_embeddings(inputs, **kwargs)

        # Mixed modality: encode each modality separately, fuse by addition
        embeddings = None

        if has_text:
            text_emb = self.get_text_embeddings(inputs, **kwargs)
            embeddings = text_emb

        if has_image:
            image_emb = self.get_image_embeddings(inputs, **kwargs)
            embeddings = image_emb if embeddings is None else embeddings + image_emb

        if has_video:
            video_emb = self.get_video_embeddings(inputs, **kwargs)
            embeddings = video_emb if embeddings is None else embeddings + video_emb

        if has_audio:
            audio_emb = self.get_audio_embeddings(inputs, **kwargs)
            embeddings = audio_emb if embeddings is None else embeddings + audio_emb

        if embeddings is not None:
            return embeddings

        raise ValueError(
            f"No supported modality found in dataset features: {list(features.keys())}"
        )


# --- Model Metadata ---

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
    license="cc-by-nc-sa-4.0",
    open_weights=True,
    public_training_code="https://github.com/encord-team/ebind",
    public_training_data=None,
    framework=["PyTorch", "safetensors"],
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets=set(),
    citation=_EBIND_CITATION,
)

# Parameter counts from the paper (arxiv:2511.14229):
#   PE-text: 353M, PE-vision: 317M, ImageBind-audio: 90M,
#   Uni3D: 1.02B, projectors: 4.2M each (audio + points).
# n_parameters is the total loaded at runtime including backbone encoders.
# memory_usage_mb assumes FP32 (4 bytes per parameter).

ebind_full = ModelMeta(
    loader=EBindWrapper,
    name="encord-team/ebind-full",
    revision="main",
    n_parameters=1_790_000_000,
    memory_usage_mb=6825,
    embed_dim=1024,
    modalities=["text", "image", "audio", "video"],
    reference="https://huggingface.co/encord-team/ebind-full",
    **_EBIND_COMMON,
)

ebind_audio_vision = ModelMeta(
    loader=EBindWrapper,
    name="encord-team/ebind-audio-vision",
    revision="main",
    n_parameters=764_200_000,
    memory_usage_mb=2916,
    embed_dim=1024,
    modalities=["text", "image", "audio", "video"],
    reference="https://huggingface.co/encord-team/ebind-audio-vision",
    **_EBIND_COMMON,
)

ebind_points_vision = ModelMeta(
    loader=EBindWrapper,
    name="encord-team/ebind-points-vision",
    revision="main",
    n_parameters=1_694_200_000,
    memory_usage_mb=6459,
    embed_dim=1024,
    modalities=["text", "image", "video"],
    reference="https://huggingface.co/encord-team/ebind-points-vision",
    **_EBIND_COMMON,
)
