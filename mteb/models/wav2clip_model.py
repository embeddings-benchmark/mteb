from __future__ import annotations

from collections.abc import Iterable
from functools import partial
from typing import Any

import numpy as np
import torch
import torchaudio
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from mteb.encoder_interface import AudioBatch, AudioData, PromptType
from mteb.model_meta import ModelMeta
from mteb.requires_package import requires_package


class Wav2ClipZeroShotWrapper:
    def __init__(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs: Any,
    ):
        requires_package(self, "wav2clip", "pip install 'mteb[wav2clip]'")
        from wav2clip import embed_audio, get_model

        self.embed_audio = embed_audio
        # audio side
        self.device = device
        self.audio_model = get_model().to(device)
        self.sampling_rate = 16_000

        # text side (CLIP)
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.clip_processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32"
        )

    def _handle_batch(
        self, batch: AudioData | Iterable[tuple[AudioData, str]]
    ) -> list[torch.Tensor]:
        waveforms: list[torch.Tensor] = []

        if isinstance(batch, tuple):  # Handle (audio, metadata) tuples
            items = [batch]
        else:
            items = batch

        for item in items:
            # dict with array and sampling_rate
            if isinstance(item, dict) and "array" in item:
                audio = item["array"]
                if isinstance(audio, np.ndarray):
                    tensor = torch.from_numpy(audio)
                elif isinstance(audio, list):
                    tensor = torch.tensor(audio, dtype=torch.float32)
                else:
                    tensor = audio  # assume it's already a torch.Tensor
                tensor = tensor.float().squeeze()
                if item.get("sampling_rate", self.sampling_rate) != self.sampling_rate:
                    resampler = torchaudio.transforms.Resample(
                        item["sampling_rate"], self.sampling_rate
                    )
                    tensor = resampler(tensor)
                
                # Apply audio truncation (30 seconds max)
                max_length = 30 * self.sampling_rate  # 30 seconds
                if tensor.shape[-1] > max_length:
                    tensor = tensor[..., :max_length]
                    
                waveforms.append(tensor)

            # dict with path
            elif isinstance(item, dict) and "path" in item:
                waveform, sr = torchaudio.load(item["path"])
                tensor = waveform.float().squeeze()
                if sr != self.sampling_rate:
                    resampler = torchaudio.transforms.Resample(sr, self.sampling_rate)
                    tensor = resampler(tensor)
                waveforms.append(tensor)

            # direct numpy or torch
            elif isinstance(item, (np.ndarray, torch.Tensor, list)):
                if isinstance(item, np.ndarray):
                    tensor = torch.from_numpy(item)
                elif isinstance(item, list):
                    tensor = torch.tensor(item, dtype=torch.float32)
                else:
                    tensor = item
                waveforms.append(tensor.float().squeeze())

            # file path string
            elif isinstance(item, str):
                waveform, sr = torchaudio.load(item)
                tensor = waveform.float().squeeze()
                if sr != self.sampling_rate:
                    resampler = torchaudio.transforms.Resample(sr, self.sampling_rate)
                    tensor = resampler(tensor)
                waveforms.append(tensor)

        return waveforms

    def get_audio_embeddings(
        self,
        audio: AudioBatch,
        *,
        show_progress_bar: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        all_embeddings = []

        if isinstance(audio, DataLoader):
            # Process each batch separately
            for batch in tqdm(audio, desc="Processing audio batches", disable=not show_progress_bar):
                batch_embeddings = []

                # Process each item in the batch individually
                wavs = self._handle_batch(batch)
                for wav in wavs:
                    # Process one audio at a time to avoid memory issues
                    wav_np = wav.unsqueeze(0).cpu().numpy()  # Add batch dimension
                    embed = self.embed_audio(wav_np, self.audio_model)

                    # Normalize
                    norm = np.linalg.norm(embed, axis=-1, keepdims=True)
                    normalized_embed = embed / norm
                    batch_embeddings.append(normalized_embed)

                all_embeddings.extend(batch_embeddings)

            return np.vstack(all_embeddings)
        else:
            # Process single batch - still do it item by item
            wavs = self._handle_batch(audio)
            for wav in wavs:
                # Process one audio at a time
                wav_np = wav.unsqueeze(0).cpu().numpy()  # Add batch dimension
                embed = self.wav2clip.embed_audio(wav_np, self.audio_model)

                # Normalize
                norm = np.linalg.norm(embed, axis=-1, keepdims=True)
                normalized_embed = embed / norm
                all_embeddings.append(normalized_embed)

            return np.vstack(all_embeddings)

    def get_text_embeddings(
        self,
        texts: list[str],
        **kwargs: Any,
    ) -> np.ndarray:
        inputs = self.clip_processor(text=texts, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            text_features = self.clip.get_text_features(**inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return text_features.cpu().numpy()

    def encode(
        self,
        inputs: list[str],
        *,
        task_name: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        return self.get_text_embeddings(inputs, **kwargs)


wav2clip_zero = ModelMeta(
    loader=partial(Wav2ClipZeroShotWrapper),
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
    training_datasets={
        # "AudioSet": ["https://research.google.com/audioset/"],
        # "FreeSound": ["https://freesound.org/"],
        # "BBC Sound Effects": ["https://sound-effects.bbcrewind.co.uk/"],
    },
)
