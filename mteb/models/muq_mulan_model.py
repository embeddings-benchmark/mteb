from __future__ import annotations

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


class MuQMuLanWrapper:
    def __init__(
        self,
        model_name: str = "OpenMuQ/MuQ-MuLan-large",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs: Any,
    ):
        requires_package(self, "muq", "pip install 'mteb[muq]'")
        from muq import MuQMuLan

        self.model_name = model_name
        self.device = device
        self.target_sampling_rate = 24000

        # Load the model
        self.model = MuQMuLan.from_pretrained(model_name).eval().to(self.device)

    def _process_audio(self, audio: AudioBatch) -> list[torch.Tensor]:
        """Process audio batch and return list of tensors."""
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
        """Handle a single batch of audio data."""
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
                        if isinstance(audio, np.ndarray):
                            waveform = torch.from_numpy(audio).float()
                        elif isinstance(audio, list):
                            waveform = torch.tensor(audio, dtype=torch.float32)
                        else:
                            waveform = (
                                audio.float()
                            )  # assume it's already a torch.Tensor
                        # Resample if needed
                        if (
                            item.get("sampling_rate", self.target_sampling_rate)
                            != self.target_sampling_rate
                        ):
                            resampler = torchaudio.transforms.Resample(
                                item["sampling_rate"], self.target_sampling_rate
                            )
                            waveform = resampler(waveform)
                        waveforms.append(self._convert_audio(waveform))
                    elif "path" in item:
                        waveforms.append(self._load_audio_file(item["path"]))
                elif isinstance(item, (np.ndarray, torch.Tensor)):
                    waveforms.append(self._convert_audio(item))
                elif isinstance(item, str):
                    waveforms.append(self._load_audio_file(item))

        return waveforms

    def _convert_audio(self, audio: AudioData) -> torch.Tensor:
        """Convert audio data to torch tensor."""
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio)
        return audio.squeeze().float()  # Ensure float32

    def _load_audio_file(self, path: str) -> torch.Tensor:
        """Load audio file and resample to target sampling rate."""
        waveform, sample_rate = torchaudio.load(path)
        waveform = waveform.float()  # Ensure float32

        if sample_rate != self.target_sampling_rate:
            resampler = torchaudio.transforms.Resample(
                sample_rate, self.target_sampling_rate
            )
            waveform = resampler(waveform)

        return waveform.squeeze()

    def get_audio_embeddings(
        self,
        audio: AudioBatch,
        **kwargs: Any,
    ) -> np.ndarray:
        """Get audio embeddings using MuQ-MuLan."""
        all_features = []

        if isinstance(audio, DataLoader):
            # Process all batches
            for batch in tqdm(audio, desc="Processing audio batches"):
                batch_features = []

                # Process each item in the batch
                for item in batch:
                    if isinstance(item, torch.Tensor):
                        waveform = item
                    elif isinstance(item, dict) and "array" in item:
                        audio = item["array"]
                        if isinstance(audio, np.ndarray):
                            waveform = torch.from_numpy(audio).float()
                        elif isinstance(audio, list):
                            waveform = torch.tensor(audio, dtype=torch.float32)
                        else:
                            waveform = (
                                audio.float()
                            )  # assume it's already a torch.Tensor
                        # Resample if needed
                        if (
                            item.get("sampling_rate", self.target_sampling_rate)
                            != self.target_sampling_rate
                        ):
                            resampler = torchaudio.transforms.Resample(
                                item["sampling_rate"], self.target_sampling_rate
                            )
                            waveform = resampler(waveform)
                    else:
                        continue

                    # Add batch dimension and move to device
                    wavs = waveform.unsqueeze(0).to(self.device)

                    with torch.no_grad():
                        audio_embeds = self.model(wavs=wavs)
                        batch_features.append(audio_embeds.cpu().numpy())

                all_features.extend(batch_features)

            return np.vstack(all_features)
        else:
            # Process single batch
            batch_features = []

            for item in audio:
                if isinstance(item, dict) and "array" in item:
                    audio = item["array"]
                    if isinstance(audio, np.ndarray):
                        waveform = torch.from_numpy(audio).float()
                    elif isinstance(audio, list):
                        waveform = torch.tensor(audio, dtype=torch.float32)
                    else:
                        waveform = audio.float()  # assume it's already a torch.Tensor
                    # Resample if needed
                    if (
                        item.get("sampling_rate", self.target_sampling_rate)
                        != self.target_sampling_rate
                    ):
                        resampler = torchaudio.transforms.Resample(
                            item["sampling_rate"], self.target_sampling_rate
                        )
                        waveform = resampler(waveform)
                elif isinstance(item, torch.Tensor):
                    waveform = item.float()
                else:
                    continue

                # Add batch dimension and move to device
                wavs = waveform.unsqueeze(0).to(self.device)

                with torch.no_grad():
                    audio_embeds = self.model(wavs=wavs)
                    batch_features.append(audio_embeds.cpu().numpy())

            return np.vstack(batch_features)

    def get_text_embeddings(
        self,
        texts: list[str],
        **kwargs: Any,
    ) -> np.ndarray:
        """Get text embeddings using MuQ-MuLan."""
        with torch.no_grad():
            text_embeds = self.model(texts=texts)

        return text_embeds.cpu().numpy()

    def encode(
        self,
        inputs: AudioBatch | list[str],
        *,
        task_name: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Encode inputs (audio or text) into embeddings."""
        if isinstance(inputs[0], str):
            return self.get_text_embeddings(inputs)
        return self.get_audio_embeddings(inputs)

    def calc_similarity(
        self, audio_embeds: np.ndarray, text_embeds: np.ndarray
    ) -> np.ndarray:
        """Calculate similarity between audio and text embeddings."""
        audio_tensor = torch.from_numpy(audio_embeds).to(self.device)
        text_tensor = torch.from_numpy(text_embeds).to(self.device)

        with torch.no_grad():
            similarity = self.model.calc_similarity(audio_tensor, text_tensor)

        return similarity.cpu().numpy()


muq_mulan_large = ModelMeta(
    loader=partial(MuQMuLanWrapper, model_name="OpenMuQ/MuQ-MuLan-large"),
    name="OpenMuQ/MuQ-MuLan-large",
    languages=["eng-Latn", "zho-Hans"],  # English and Chinese support
    revision="8a081dbcf84edd47ea7db3c4ecb8fd1ec1ddacfe8a081dbcf84edd47ea7db3c4ecb8fd1ec1ddacfe",
    release_date="2025-01-01",
    modalities=["audio", "text"],
    n_parameters=630_000_000,
    memory_usage_mb=2530,
    max_tokens=None,
    embed_dim=512,
    license="CC-BY-NC-4.0",
    open_weights=True,
    public_training_code="https://github.com/tencent-ailab/MuQ",
    public_training_data="https://github.com/tencent-ailab/MuQ",
    framework=["PyTorch"],
    reference="https://huggingface.co/OpenMuQ/MuQ-MuLan-large",
    similarity_fn_name="cosine",
    use_instructions=False,
    training_datasets={},
)
