from __future__ import annotations

from collections.abc import Iterable
from functools import partial
from typing import Any

import numpy as np
import torch
import torchaudio
import wav2clip
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
        requires_package(
            self,
            "wav2clip",
            "pip install wav2clip"
        )
        # audio side
        self.device = device
        self.audio_model = wav2clip.get_model().to(device)
        # wav2clip defaults to 16_000 Hz
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
                tensor = (
                    torch.from_numpy(audio)
                    if isinstance(audio, np.ndarray)
                    else item["array"]
                )
                tensor = tensor.float().squeeze()
                if item.get("sampling_rate", self.sampling_rate) != self.sampling_rate:
                    resampler = torchaudio.transforms.Resample(
                        item["sampling_rate"], self.sampling_rate
                    )
                    tensor = resampler(tensor)
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
            elif isinstance(item, (np.ndarray, torch.Tensor)):
                tensor = (
                    torch.from_numpy(item) if isinstance(item, np.ndarray) else item
                )
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
        **kwargs: Any,
    ) -> np.ndarray:
        # collect all waveforms
        if isinstance(audio, DataLoader):
            wavs: list[torch.Tensor] = []
            for batch in tqdm(audio, desc="Preparing audio for wav2clip"):
                wavs.extend(self._handle_batch(batch))
        else:
            wavs = self._handle_batch(audio)

        # embed with wav2clip
        embeds = wav2clip.embed_audio(wavs, self.audio_model)
        # L2 normalize
        norms = np.linalg.norm(embeds, axis=-1, keepdims=True)
        return embeds / norms

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
        inputs: AudioBatch | list[str],
        *,
        task_name: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        if isinstance(inputs[0], str):
            return self.get_text_embeddings(inputs)
        return self.get_audio_embeddings(inputs)


# register for MTEB
wav2clip_zero = ModelMeta(
    loader=partial(Wav2ClipZeroShotWrapper),
    name="lyrebird/wav2clip",
    languages=["eng-Latn"],
    revision="N/A",
    release_date="2023-01-01",
    modalities=["audio", "text"],
    n_parameters=0,
    memory_usage_mb=0,
    max_tokens=float("inf"),
    embed_dim=512,
    license="mit",
    open_weights=True,
    framework=["PyTorch"],
    reference="https://github.com/andabi/wav2clip",
    similarity_fn_name="cosine",
    use_instructions=False,
    public_training_code="https://github.com/LAION-AI/CLAP",
    public_training_data="https://laion.ai/blog/laion-audio-630k/",
    training_datasets={},
)
