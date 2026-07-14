from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoModel

from mteb.models import ModelMeta
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.modality_collators import AudioCollator, VideoCollator

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from mteb import TaskMetadata
    from mteb.types import Array, BatchedInput, PromptType
    from mteb.types._encoder_io import AudioInput, TextInput


class FusionEmbeddingWrapper(AbsEncoder):
    """fusion-embedding-1: a unified text/image/video/audio embedding model.

    This wrapper exposes all four encoding paths. Audio: frozen Qwen2.5-Omni audio
    tower -> trained 16.4M-parameter perceiver-resampler -> tokens spliced into the
    frozen Qwen3-VL-Embedding-2B input stream -> last-token pooling -> Matryoshka
    truncation. Text: the base model's chat template -> last-token pooling ->
    diagonal whitening -> Matryoshka truncation. Image and video: the frozen base
    model's own encode paths (nothing trained touches them; the base's published
    behavior is inherited unchanged). Video preprocessing natively follows the base's
    reference pipeline (decoded frame tensors, up to 64 uniformly sampled frames). Inputs
    combining several modalities are embedded per modality and summed elementwise.
    The base model is byte-identical to its original release; only the connector
    was trained (audio-text contrastive).

    The model loads through ``AutoModel`` with ``trust_remote_code``; the remote code
    downloads the frozen base and audio tower from their original repositories and only
    the trained connector weights from this repository.
    """

    sampling_rate = 16_000
    max_text_tokens = 254
    video_num_frames = 64  # the base's reference max_frames; sampled uniformly

    def __init__(
        self,
        model_name: str,
        revision: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs: Any,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.model = AutoModel.from_pretrained(
            model_name,
            revision=revision,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()

    def get_audio_embeddings(
        self,
        inputs: DataLoader[AudioInput],
        show_progress_bar: bool = True,
        **kwargs: Any,
    ) -> Array:
        embeddings = []

        for batch in tqdm(inputs, disable=not show_progress_bar):
            waveforms = []
            for audio in batch["audio"]:
                array = np.asarray(audio["array"], dtype=np.float32)
                if array.ndim > 1:
                    array = array.mean(axis=-1)
                waveforms.append(array)

            with torch.no_grad():
                batch_embeddings = self.model.embed_audio_batch(
                    waveforms, sr=self.sampling_rate
                )

            embeddings.append(batch_embeddings.float().cpu().numpy())

        return np.vstack(embeddings)

    def get_video_embeddings(
        self,
        inputs: DataLoader[BatchedInput],
        show_progress_bar: bool = True,
        **kwargs: Any,
    ) -> Array:
        # The video path is the frozen base model's own encode path, exposed by
        # the remote code. VideoCollator supplies decoded [T, C, H, W] uint8
        # frame tensors (torchcodec), which embed_video consumes directly.
        embeddings = []

        for batch in tqdm(inputs, disable=not show_progress_bar, desc="Video Encoding"):
            with torch.no_grad():
                batch_embeddings = torch.stack(
                    [
                        self.model.embed_video(video, max_frames=self.video_num_frames)
                        for video in batch["video"]
                    ]
                )

            embeddings.append(batch_embeddings.float().cpu().numpy())

        return np.vstack(embeddings)

    def get_image_embeddings(
        self,
        inputs: DataLoader[BatchedInput],
        show_progress_bar: bool = True,
        **kwargs: Any,
    ) -> Array:
        # The image path is the frozen base model's own encode path (audio-side
        # components never touch it), exposed by the remote code as a single-image
        # method; images are embedded one at a time (as in other merged wrappers).
        embeddings = []

        for batch in tqdm(inputs, disable=not show_progress_bar, desc="Image Encoding"):
            with torch.no_grad():
                batch_embeddings = torch.stack(
                    [self.model.embed_image(image) for image in batch["image"]]
                )

            embeddings.append(batch_embeddings.float().cpu().numpy())

        return np.vstack(embeddings)

    def get_text_embeddings(
        self,
        inputs: DataLoader[TextInput],
        show_progress_bar: bool = True,
        **kwargs: Any,
    ) -> Array:
        embeddings = []

        for batch in tqdm(inputs, disable=not show_progress_bar):
            with torch.no_grad():
                batch_embeddings = self.model.embed_text_batch(
                    batch["text"], max_tokens=self.max_text_tokens
                )

            embeddings.append(batch_embeddings.float().cpu().numpy())

        return np.vstack(embeddings)

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
        present = [m for m in ("audio", "video", "image", "text") if m in features]
        if not present:
            raise ValueError(
                "fusion-embedding supports audio, video, image, and text inputs, "
                f"got: {list(features)}"
            )

        original_collate_fn = inputs.collate_fn
        embeddings = None
        for modality in present:
            if modality in {"audio", "video"} and "video" in present:
                inputs.collate_fn = VideoCollator(
                    target_sampling_rate=self.sampling_rate,
                    num_frames=self.video_num_frames,
                )
            elif modality == "audio":
                inputs.collate_fn = AudioCollator(
                    target_sampling_rate=self.sampling_rate
                )
            else:
                inputs.collate_fn = original_collate_fn

            if modality == "audio":
                modality_embeddings = self.get_audio_embeddings(inputs, **kwargs)
            elif modality == "video":
                modality_embeddings = self.get_video_embeddings(inputs, **kwargs)
            elif modality == "image":
                modality_embeddings = self.get_image_embeddings(inputs, **kwargs)
            else:
                modality_embeddings = self.get_text_embeddings(inputs, **kwargs)

            if embeddings is None:
                embeddings = modality_embeddings
            else:
                # Fused inputs: elementwise sum across modalities, as in the
                # CLIP wrapper's text+image handling.
                if len(modality_embeddings) != len(embeddings):
                    raise ValueError(
                        "all modalities must have the same length for fused embeddings"
                    )
                embeddings += modality_embeddings
        return embeddings


fusion_embedding_1_2b_preview = ModelMeta(
    loader=FusionEmbeddingWrapper,
    name="EximiusLabs/fusion-embedding-1-2b-preview",
    languages=["eng-Latn"],
    open_weights=True,
    revision="b551ea8033bee3cd51468cbde2bb25397292e0b3",
    release_date="2026-07-06",
    modalities=["audio", "image", "text", "video"],
    n_parameters=2_800_000_000,
    n_embedding_parameters=311_164_928,
    memory_usage_mb=10681,
    max_tokens=254,
    embed_dim=[2048, 1536, 1024, 512, 256, 128, 64],
    license="cc-by-nc-4.0",
    reference="https://huggingface.co/EximiusLabs/fusion-embedding-1-2b-preview",
    similarity_fn_name="cosine",
    framework=["PyTorch", "Transformers", "safetensors"],
    use_instructions=True,
    public_training_code="https://github.com/Eximius-Labs/fusion-embedding",
    public_training_data=None,
    training_datasets={
        # AudioCaps (train split)
        "AudioCapsA2TRetrieval",
        "AudioCapsT2ARetrieval",
        "AudioCapsAVA2VRetrieval",
        "AudioCapsAVAT2VRetrieval",
        "AudioCapsAVT2VRetrieval",
        "AudioCapsAVT2VARetrieval",
        "AudioCapsAVV2ARetrieval",
        "AudioCapsAVV2TRetrieval",
        "AudioCapsAVVA2TRetrieval",
        "AudioCapsAVVT2ARetrieval",
        # FSD50K (dev split); 13.6% of FSDKaggle2019 test clips appear in FSD50K dev
        "FSD50K",
        "FSD2019Kaggle",
        # AudioSet-SL / WavCaps captions over strongly-labelled AudioSet clips
        "AudioSet",
        "AudioSetMini",
        # LAION-FreeSound (not in MTEB)
        # Clotho / ESC-50 / UrbanSound8K / VGGSound eval clips excluded from training
        # by id blacklist at ingestion (see the model card)
    },
    model_type=["dense"],
    citation="""
@software{fusion_embedding_2026,
  title  = {Fusion Embedding 1: A Unified Embedding Space for Text,
            Image, Video, and Audio},
  author = {Tonmoy, Abdul Basit},
  year   = {2026},
  url    = {https://github.com/Eximius-Labs/fusion-embedding}
}
""",
    extra_requirements_groups=["fusion-embedding"],
)
