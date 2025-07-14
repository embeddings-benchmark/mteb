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
        model_name: str = "microsoft/msclap",
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
        self.sampling_rate = 48000  # CLAP's expected sampling rate

        if "2022" in self.model_name:
            self.version = "2022"
        elif "clapcap" in self.model_name:
            self.version = "clapcap"
        else:
            self.version = "2023"
        self.version = "2023"
        self.use_cuda = device == "cuda"
        self.device = device
        self.model = CLAP(version=self.version, use_cuda=self.use_cuda)

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
                        if item.get("sampling_rate", self.sampling_rate) != self.sampling_rate:
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

    def get_audio_embeddings(
        self,
        audio: AudioBatch,
        **kwargs: Any,
    ) -> np.ndarray:
        all_features = []

        if isinstance(audio, DataLoader):
            # Process all batches
            for batch in tqdm(audio, desc="Processing audio batches"):
                batch_features = []
                # Process each item in the batch individually to avoid memory issues
                for item in batch:
                    if isinstance(item, torch.Tensor):
                        item = {"array": item.numpy(), "sampling_rate": self.sampling_rate}
                    elif isinstance(item, dict) and "array" in item:
                        # Ensure sampling_rate is available
                        if "sampling_rate" not in item:
                            item["sampling_rate"] = self.sampling_rate
                    
                    # Convert audio to proper format for CLAP
                    audio_data = item["array"]
                    if isinstance(audio_data, torch.Tensor):
                        audio_data = audio_data.numpy()
                    
                    with torch.no_grad():
                        audio_features = self.model.get_audio_embedding_from_data(
                            x=audio_data, 
                            use_tensor=True
                        )
                        # Normalize embeddings
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
                if isinstance(item, torch.Tensor):
                    item = {"array": item.numpy(), "sampling_rate": self.sampling_rate}
                elif isinstance(item, dict) and "array" in item:
                    # Ensure sampling_rate is available
                    if "sampling_rate" not in item:
                        item["sampling_rate"] = self.sampling_rate

                # Convert audio to proper format for CLAP
                audio_data = item["array"]
                if isinstance(audio_data, torch.Tensor):
                    audio_data = audio_data.numpy()

                with torch.no_grad():
                    audio_features = self.model.get_audio_embedding_from_data(
                        x=audio_data, 
                        use_tensor=True
                    )
                    # Normalize embeddings
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
        with torch.no_grad():
            text_features = self.model.get_text_embedding(texts, use_tensor=True)
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


# Microsoft CLAP Model metadata
ms_clap_2022 = ModelMeta(
    loader=partial(MSClapWrapper, model_name="microsoft/msclap-2022"),
    name="microsoft/msclap-2022",
    languages=["eng-Latn"],
    revision="N/A",  
    release_date="2022-12-01", 
    modalities=["audio", "text"],
    n_parameters=86_000_000,  # Estimated based on architecture
    memory_usage_mb=350,  # Estimated
    max_tokens=float("inf"),
    embed_dim=1024,  # Common embedding dimension for Microsoft CLAP
    license="mit",
    open_weights=True,
    public_training_code="https://github.com/microsoft/CLAP",
    public_training_data="Various audio datasets",  # Microsoft used multiple datasets
    framework=["PyTorch"],
    reference="https://github.com/microsoft/CLAP",
    similarity_fn_name="cosine",
    use_instructions=False,
    training_datasets={},
)

ms_clap_2023 = ModelMeta(
    loader=partial(MSClapWrapper, model_name="microsoft/msclap"),
    name="microsoft/msclap",
    languages=["eng-Latn"],
    revision="N/A",
    release_date="2023-09-01",  # Based on arXiv paper date
    modalities=["audio", "text"],
    n_parameters=125_000_000,  # Estimated - 2023 version is larger
    memory_usage_mb=480,
    max_tokens=float("inf"),
    embed_dim=1024,
    license="mit",
    open_weights=True,
    public_training_code="https://github.com/microsoft/CLAP",
    public_training_data="Various audio datasets",
    framework=["PyTorch"],
    reference="https://github.com/microsoft/CLAP",
    similarity_fn_name="cosine",
    use_instructions=False,
    training_datasets={
        # Enhanced training with more diverse datasets
    },
)

ms_clap_clapcap = ModelMeta(
    loader=partial(MSClapWrapper, model_name="microsoft/msclap-clapcap"),
    name="microsoft/clap-clapcap",
    languages=["eng-Latn"],
    revision="N/A",
    release_date="2023-09-01",
    modalities=["audio", "text"],
    n_parameters=125_000_000, 
    memory_usage_mb=480,
    max_tokens=float("inf"),
    embed_dim=1024,
    license="mit",
    open_weights=True,
    public_training_code="https://github.com/microsoft/CLAP",
    public_training_data="Various audio datasets + captioning data",
    framework=["PyTorch"],
    reference="https://github.com/microsoft/CLAP",
    similarity_fn_name="cosine",
    use_instructions=False,
    training_datasets={},
)