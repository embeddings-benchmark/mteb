from __future__ import annotations

import logging
from collections.abc import Iterable
from functools import partial
from typing import Any

import numpy as np
import torch
import torchaudio
from torch.utils.data import DataLoader
from tqdm import tqdm

from mteb.encoder_interface import AudioBatch, AudioData, PromptType
from mteb.model_meta import ModelMeta
from mteb.requires_package import requires_package

logger = logging.getLogger(__name__)


class MSClapWrapper:
    def __init__(
        self,
        model_name: str = "microsoft/msclap-2023",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs: Any,
    ):
        requires_package(
            self,
            "msclap",
            "pip install 'mteb[msclap]'",
        )
        from msclap import CLAP

        self.model_name = model_name
        self.sampling_rate = 48000

        if "2022" in self.model_name:
            self.version = "2022"
        elif "2023" in self.model_name:
            self.version = "2023"
        else:
            self.version = "2023"
        self.use_cuda = device == "cuda"
        self.device = device
        self.model = CLAP(version=self.version, use_cuda=self.use_cuda)
        self.model.clap = self.model.clap.to(self.device)

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
                        if (
                            item.get("sampling_rate", self.sampling_rate)
                            != self.sampling_rate
                        ):
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
        return audio.squeeze().float()

    def _load_audio_file(self, path: str) -> torch.Tensor:
        waveform, sample_rate = torchaudio.load(path)
        waveform = waveform.float()  # Ensure float32
        if sample_rate != self.sampling_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.sampling_rate)
            waveform = resampler(waveform)
        return waveform.squeeze()

    def _process_audio_to_tensor(self, audio_item) -> torch.Tensor:
        """Convert various audio formats to torch tensor with proper sampling rate"""
        if isinstance(audio_item, dict):
            if "array" in audio_item:
                audio = audio_item["array"]
                sr = audio_item.get("sampling_rate", self.sampling_rate)

                if isinstance(audio, np.ndarray):
                    audio = torch.from_numpy(audio).float()
                elif isinstance(audio, list):
                    audio = torch.tensor(audio, dtype=torch.float32)
                else:
                    audio = audio.float()

                # Resample if needed
                if sr != self.sampling_rate:
                    resampler = torchaudio.transforms.Resample(sr, self.sampling_rate)
                    audio = resampler(audio)

                # Apply audio truncation (30 seconds max)
                max_length = 30 * self.sampling_rate  # 30 seconds
                if audio.shape[-1] > max_length:
                    audio = audio[..., :max_length]

                return audio.squeeze()

            elif "path" in audio_item:
                return self._load_audio_file(audio_item["path"])

        elif isinstance(audio_item, (np.ndarray, torch.Tensor)):
            if isinstance(audio_item, np.ndarray):
                audio_item = torch.from_numpy(audio_item)
            return audio_item.squeeze().float()

        elif isinstance(audio_item, str):
            return self._load_audio_file(audio_item)

        else:
            raise ValueError(f"Unsupported audio format: {type(audio_item)}")

    def _load_audio_file(self, path: str) -> torch.Tensor:
        """Load audio file and convert to proper format"""
        waveform, sample_rate = torchaudio.load(path)
        waveform = waveform.float()

        if sample_rate != self.sampling_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.sampling_rate)
            waveform = resampler(waveform)

        return waveform.squeeze()

    def get_audio_embeddings(
        self,
        audio: AudioBatch,
        *,
        show_progress_bar: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        """Get audio embeddings using direct tensor processing (no temp files)"""
        all_features = []

        if isinstance(audio, DataLoader):
            # Process all batches
            for batch in tqdm(audio, desc="Processing audio batches", disable=not show_progress_bar):
                batch_features = self._process_audio_batch(batch)
                all_features.extend(batch_features)
        else:
            # Process single batch
            batch_features = self._process_audio_batch(audio)
            all_features.extend(batch_features)

        return np.vstack(all_features)

    def _process_audio_batch(self, batch) -> list[np.ndarray]:
        """Process a batch of audio items and return embeddings"""
        batch_features = []

        for item in batch:
            # Convert to tensor
            audio_tensor = self._process_audio_to_tensor(item)

            # Ensure it's in the right format [batch_size, samples]
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)  # Add batch dimension

            audio_tensor = audio_tensor.to(self.device)
            # Get embeddings using the internal audio encoder
            with torch.no_grad():
                # Use the internal method: [0] the audio emebdding, [1] has output class probabilities
                audio_features = self.model.clap.audio_encoder(audio_tensor)[0]

                # Normalize embeddings
                audio_features = audio_features / audio_features.norm(
                    dim=-1, keepdim=True
                )

                batch_features.append(audio_features.cpu().numpy())

        return batch_features

    def get_text_embeddings(
        self,
        texts: list[str],
        **kwargs: Any,
    ) -> np.ndarray:
        with torch.no_grad():
            preprocessed_texts = self.model.preprocess_text(texts)
            if isinstance(preprocessed_texts, dict):
                preprocessed_texts = {
                    k: v.to(self.device) for k, v in preprocessed_texts.items()
                }
            else:
                preprocessed_texts = preprocessed_texts.to(self.device)
            text_features = self.model.clap.caption_encoder(preprocessed_texts)
            # Normalize embeddings
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


# Microsoft CLAP Model metadata
ms_clap_2022 = ModelMeta(
    loader=partial(MSClapWrapper, model_name="microsoft/msclap-2022"),
    name="microsoft/msclap-2022",
    languages=["eng-Latn"],
    revision="N/A",
    release_date="2022-12-01",
    modalities=["audio", "text"],
    n_parameters=196_000_000,
    memory_usage_mb=750,
    max_tokens=None,
    embed_dim=1024,
    license="mit",
    open_weights=True,
    public_training_code="https://github.com/microsoft/CLAP",
    public_training_data="https://github.com/microsoft/CLAP",
    framework=["PyTorch"],
    reference="https://github.com/microsoft/CLAP",
    similarity_fn_name="cosine",
    use_instructions=False,
    training_datasets={},
)

ms_clap_2023 = ModelMeta(
    loader=partial(MSClapWrapper, model_name="microsoft/msclap-2023"),
    name="microsoft/msclap-2023",
    languages=["eng-Latn"],
    revision="N/A",
    release_date="2023-09-01",
    modalities=["audio", "text"],
    n_parameters=160_000_000,
    memory_usage_mb=610,
    max_tokens=None,
    embed_dim=1024,
    license="mit",
    open_weights=True,
    public_training_code="https://github.com/microsoft/CLAP",
    public_training_data="https://github.com/microsoft/CLAP",
    framework=["PyTorch"],
    reference="https://github.com/microsoft/CLAP",
    similarity_fn_name="cosine",
    use_instructions=False,
    training_datasets={},
)
