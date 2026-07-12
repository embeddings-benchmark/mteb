from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoModel

from mteb.models import ModelMeta
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.modality_collators import AudioCollator

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from mteb import TaskMetadata
    from mteb.types import Array, BatchedInput, PromptType
    from mteb.types._encoder_io import AudioInput, TextInput


class FusionEmbeddingWrapper(AbsEncoder):
    """fusion-embedding-1: a unified text/image/video/audio embedding model.

    This wrapper exposes the audio and text encoding paths (the modalities MAEB
    exercises). Audio: frozen Qwen2.5-Omni audio tower -> trained 16.4M-parameter
    perceiver-resampler -> tokens spliced into the frozen Qwen3-VL-Embedding-2B input
    stream -> last-token pooling -> Matryoshka truncation. Text: the base model's chat
    template -> last-token pooling -> diagonal whitening -> Matryoshka truncation. The
    base model is byte-identical to its original release; only the connector was
    trained (audio-text contrastive).

    The model loads through ``AutoModel`` with ``trust_remote_code``; the remote code
    downloads the frozen base and audio tower from their original repositories and only
    the trained connector weights from this repository.
    """

    sampling_rate = 16_000
    max_text_tokens = 254

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
        if "audio" in inputs.dataset.features:
            inputs.collate_fn = AudioCollator(target_sampling_rate=self.sampling_rate)
            return self.get_audio_embeddings(inputs, **kwargs)
        if "text" in inputs.dataset.features:
            return self.get_text_embeddings(inputs, **kwargs)
        raise ValueError(
            "fusion-embedding supports audio and text inputs, got: "
            f"{list(inputs.dataset.features)}"
        )


fusion_embedding_1_2b_preview = ModelMeta(
    loader=FusionEmbeddingWrapper,
    name="EximiusLabs/fusion-embedding-1-2b-preview",
    languages=["eng-Latn"],
    open_weights=True,
    revision="e6d91bc06920e74553b5ea52244ebdf7d1a82402",
    release_date="2026-07-06",
    modalities=["audio", "text"],
    n_parameters=2_800_000_000,
    n_embedding_parameters=None,
    memory_usage_mb=10681,  # Calculated using model.calculate_memory_usage_mb()
    max_tokens=254,
    embed_dim=[2048, 1536, 1024, 512, 256, 128, 64],  # Matryoshka ladder
    license="cc-by-nc-4.0",
    reference="https://huggingface.co/EximiusLabs/fusion-embedding-1-2b-preview",
    similarity_fn_name="cosine",
    framework=["PyTorch", "Transformers", "safetensors"],
    use_instructions=True,
    public_training_code="https://github.com/Eximius-Labs/fusion-embedding-1",
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
  url    = {https://github.com/Eximius-Labs/fusion-embedding-1}
}
""",
    extra_requirements_groups=["audio", "fusion-embedding"],
)
