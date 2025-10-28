import warnings
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from mteb import TaskMetadata
from mteb._requires_package import requires_package
from mteb.models import ModelMeta
from mteb.types import Array, BatchedInput, PromptType
from mteb.types._encoder_io import AudioInput, TextInput


class Wav2ClipZeroShotWrapper:
    def __init__(
        self,
        model_name: str,
        revision: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_audio_length_s: float = 30.0,
        **kwargs: Any,
    ):
        requires_package(self, "wav2clip", "pip install 'mteb[wav2clip]'")
        from wav2clip import embed_audio, get_model

        self.embed_audio = embed_audio
        # audio side
        self.device = device
        self.audio_model = get_model().to(device)
        self.sampling_rate = 16_000
        self.max_audio_length_s = max_audio_length_s

        # text side (CLIP)
        self.clip = CLIPModel.from_pretrained(model_name, revision=revision).to(device)
        self.clip.eval()
        self.clip_processor = CLIPProcessor.from_pretrained(
            model_name, revision=revision
        )

    def get_audio_embeddings(
        self,
        inputs: DataLoader[AudioInput],
        show_progress_bar: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        import torchaudio

        all_embeddings = []

        # Process each DataLoader batch separately
        for batch in tqdm(
            inputs, desc="Processing audio batches", disable=not show_progress_bar
        ):
            audio_arrays = []
            for a in batch["audio"]:
                array = torch.tensor(a["array"], dtype=torch.float32)
                sr = a.get("sampling_rate", None)
                if sr is None:
                    warnings.warn(
                        f"No sampling_rate provided for an audio sample. "
                        f"Assuming {self.sampling_rate} Hz (model default)."
                    )
                    sr = self.sampling_rate

                if sr != self.sampling_rate:
                    resampler = torchaudio.transforms.Resample(
                        orig_freq=sr, new_freq=self.sampling_rate
                    )
                    array = resampler(array)
                audio_arrays.append(array.numpy())

            max_length = max(wav.shape[-1] for wav in audio_arrays)
            padded_wavs = []
            for wav in audio_arrays:
                if wav.shape[-1] < max_length:
                    # Pad with zeros
                    pad_length = max_length - wav.shape[-1]
                    padded_wav = torch.nn.functional.pad(wav, (0, pad_length))
                else:
                    padded_wav = wav
                padded_wavs.append(padded_wav)

            # Stack into batch tensor
            batch_tensor = torch.stack(padded_wavs).cpu().numpy()

            # Process entire batch at once
            batch_embeds = self.embed_audio(batch_tensor, self.audio_model)

            # Normalize each embedding in the batch
            norms = np.linalg.norm(batch_embeds, axis=-1, keepdims=True)
            normalized_embeds = batch_embeds / norms

            # Add each embedding from the batch
            for embed in normalized_embeds:
                all_embeddings.append(embed.reshape(1, -1))

        return np.vstack(all_embeddings)

    def get_text_embeddings(
        self,
        inputs: DataLoader[TextInput],
        show_progress_bar: bool = True,
        **kwargs: Any,
    ) -> Array:
        text_embeddings = []
        for batch in tqdm(
            inputs, disable=not show_progress_bar, desc="Processing text batches"
        ):
            texts = batch["text"]
            features = self.clip_processor(
                text=texts, return_tensors="pt", padding=True, truncation=True
            )
            features = {k: v.to(self.device) for k, v in features.items()}

            with torch.no_grad():
                text_features = self.clip.get_text_features(**features)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            text_embeddings.append(text_features.cpu().numpy())

        return np.vstack(text_embeddings)

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
        text_embeddings = None
        audio_embeddings = None

        if "text" in inputs.dataset.features:
            text_embeddings = self.get_text_embeddings(inputs, **kwargs)
        if "audio" in inputs.dataset.features:
            audio_embeddings = self.get_audio_embeddings(inputs, **kwargs)

        if text_embeddings is not None and audio_embeddings is not None:
            if len(text_embeddings) != len(audio_embeddings):
                raise ValueError(
                    "The number of texts and images must have the same length"
                )
            fused_embeddings = text_embeddings + audio_embeddings
            return fused_embeddings
        elif text_embeddings is not None:
            return text_embeddings
        elif audio_embeddings is not None:
            return audio_embeddings
        raise ValueError


wav2clip_zero = ModelMeta(
    loader=Wav2ClipZeroShotWrapper,
    name="lyrebird/wav2clip",
    languages=["eng-Latn"],
    revision="N/A",
    release_date="2022-03-15",
    modalities=["audio", "text"],
    n_parameters=163_000_000,  # wav2clip: 11.7M + CLIP: 151.3M ≈ 163M
    memory_usage_mb=622,  # wav2clip: 44.65MB + CLIP: 577.08MB ≈ 622MB
    max_tokens=None,
    embed_dim=512,
    license="mit",
    open_weights=True,
    framework=["PyTorch"],
    reference="https://github.com/descriptinc/lyrebird-wav2clip",
    similarity_fn_name="cosine",
    use_instructions=False,
    public_training_code="https://github.com/descriptinc/lyrebird-wav2clip",
    public_training_data="https://github.com/descriptinc/lyrebird-wav2clip#data",
    training_datasets=set(
        # "AudioSet": ["https://research.google.com/audioset/"],
        # "FreeSound": ["https://freesound.org/"],
        # "BBC Sound Effects": ["https://sound-effects.bbcrewind.co.uk/"],
    ),
)
