from __future__ import annotations

from collections.abc import Iterable
from functools import partial
from typing import Any

import numpy as np
import torch
import torchaudio
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import ClapModel, ClapProcessor, pipeline

from mteb.encoder_interface import AudioBatch, AudioData, PromptType
from mteb.model_meta import ModelMeta


class ClapZeroShotWrapper:
    def __init__(
        self,
        model_name: str = "laion/clap-htsat-unfused",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs: Any,
    ):
        self.model_name = model_name
        self.device = device
        self.model = ClapModel.from_pretrained(model_name).to(self.device)
        self.processor = ClapProcessor.from_pretrained(model_name)
        self.sampling_rate = self.processor.feature_extractor.sampling_rate

        self.pipeline = pipeline(
            task="zero-shot-audio-classification", model=model_name, device=device
        )

    def _process_audio(self, audio: AudioBatch) -> list[torch.Tensor]:
        processed_audio = []

        if isinstance(audio, DataLoader):
            for batch in audio:
                processed_audio.extend(self._handle_batch(batch))
        else:
            processed_audio = self._handle_batch(audio)

        return processed_audio

    def _handle_batch(
        self, batch: AudioData | Iterable[tuple[AudioData, str]]
    ) -> list[torch.Tensor]:
        waveforms = []

        if isinstance(batch, tuple):  # Handle (audio, metadata) tuples
            for audio, _ in batch:
                waveforms.append(self._convert_audio(audio))
        else:
            for item in batch:
                if isinstance(item, dict):
                    if "array" in item:
                        audio = item["array"]
                        # Convert to torch tensor and ensure float32
                        audio = (
                            torch.from_numpy(audio).float()
                            if isinstance(audio, np.ndarray)
                            else audio.float()
                        )
                        if item["sampling_rate"] != self.sampling_rate:
                            resampler = torchaudio.transforms.Resample(
                                item["sampling_rate"], self.sampling_rate
                            )
                            audio = resampler(audio)
                        waveforms.append(self._convert_audio(audio))
                    elif "path" in item:
                        waveforms.append(self._load_audio_file(item["path"]))
                elif isinstance(item, (np.ndarray, torch.Tensor)):
                    waveforms.append(self._convert_audio(item))
                elif isinstance(item, str):
                    waveforms.append(self._load_audio_file(item))

        return waveforms

    def _convert_audio(self, audio: AudioData) -> torch.Tensor:
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio)
        return audio.squeeze().float()  # Ensure float32

    def _load_audio_file(self, path: str) -> torch.Tensor:
        waveform, sample_rate = torchaudio.load(path)
        waveform = waveform.float()  # Ensure float32
        if sample_rate != self.sampling_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.sampling_rate)
            waveform = resampler(waveform)
        return waveform.squeeze()

    def _convert_audio(self, audio: AudioData) -> torch.Tensor:
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio)
        return audio.squeeze()

    def _load_audio_file(self, path: str) -> torch.Tensor:
        waveform, sample_rate = torchaudio.load(path)
        if sample_rate != self.sampling_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.sampling_rate)
            waveform = resampler(waveform)
        return waveform.squeeze()

    def get_audio_embeddings(
        self,
        audio: AudioBatch,
        **kwargs: Any,
    ) -> np.ndarray:
        all_features = []
        target_sampling_rate = 48000  # CLAP's expected sampling rate

        if isinstance(audio, DataLoader):
            # Process all batches
            for batch in tqdm(audio, desc="Processing audio batches"):
                batch_features = []
                # Process each item in the batch individually to avoid memory issues
                for item in batch:
                    inputs = self.pipeline.feature_extractor(
                        [item["array"]],
                        sampling_rate=target_sampling_rate,
                        return_tensors="pt",
                        padding=True,
                    )
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}

                    with torch.no_grad():
                        audio_features = self.pipeline.model.get_audio_features(
                            **inputs
                        )
                        audio_features = audio_features / audio_features.norm(
                            dim=-1, keepdim=True
                        )
                        batch_features.append(audio_features.cpu().numpy())

                all_features.extend(batch_features)

            return np.vstack(all_features)
        else:
            # Process single batch
            batch_features = []
            for item in audio:
                inputs = self.pipeline.feature_extractor(
                    [item["array"]],
                    sampling_rate=target_sampling_rate,
                    return_tensors="pt",
                    padding=True,
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    audio_features = self.pipeline.model.get_audio_features(**inputs)
                    audio_features = audio_features / audio_features.norm(
                        dim=-1, keepdim=True
                    )
                    batch_features.append(audio_features.cpu().numpy())

            return np.vstack(batch_features)

    def get_text_embeddings(
        self,
        texts: list[str],
        **kwargs: Any,
    ) -> np.ndarray:
        inputs = self.processor(text=texts, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
            # Normalize embeddings
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


# Model metadata
clap_htsat_unfused = ModelMeta(
    loader=partial(ClapZeroShotWrapper, model_name="laion/clap-htsat-unfused"),
    name="laion/clap-htsat-unfused",
    languages=["en"],
    revision="main",
    release_date="2023-05-22",
    modalities=["audio", "text"],
    n_parameters=None,  # Fill in actual number
    memory_usage_mb=None,  # Fill in actual number
    max_tokens=float("inf"),
    embed_dim=512,  # Verify this
    license="MIT",
    open_weights=True,
    public_training_code="https://github.com/LAION-AI/CLAP",
    public_training_data="LAION-Audio-630K",
    framework=["PyTorch"],
    reference="https://huggingface.co/laion/clap-htsat-unfused",
    similarity_fn_name="cosine",
    use_instructions=False,
    training_datasets={"LAION-Audio-630K": ["https://laion.ai/blog/laion-audio-630k/"]},
)
