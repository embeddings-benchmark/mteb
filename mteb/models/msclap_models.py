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
        self.device = device
        self.sampling_rate = 48000

        if "2022" in self.model_name:
            self.version = "2022"
        elif "2023" in self.model_name:
            self.version = "2023"
        else:
            self.version = "2023"

        self.use_cuda = device == "cuda"
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
                        audio_array = item["array"]
                        # Convert to torch tensor and ensure float32
                        if isinstance(audio_array, np.ndarray):
                            audio_array = torch.from_numpy(audio_array).float()
                        elif isinstance(audio_array, list):
                            # Convert list to numpy array first, then to tensor
                            audio_array = torch.from_numpy(
                                np.array(audio_array)
                            ).float()
                        else:
                            # Already a tensor
                            audio_array = audio_array.float()

                        # Handle resampling if needed
                        if (
                            "sampling_rate" in item
                            and item["sampling_rate"] != self.sampling_rate
                        ):
                            resampler = torchaudio.transforms.Resample(
                                item["sampling_rate"], self.sampling_rate
                            )
                            audio_array = resampler(audio_array)

                        # Apply audio truncation (30 seconds max)
                        max_length = 30 * self.sampling_rate  # 30 seconds
                        if audio_array.shape[-1] > max_length:
                            audio_array = audio_array[..., :max_length]

                        # Only squeeze here, don't call _convert_audio again
                        waveforms.append(audio_array.squeeze())
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
        audio = audio.squeeze().float()  # Ensure float32

        # Apply audio truncation (30 seconds max)
        max_length = 30 * self.sampling_rate  # 30 seconds
        if audio.shape[-1] > max_length:
            audio = audio[..., :max_length]

        return audio

    def _load_audio_file(self, path: str) -> torch.Tensor:
        waveform, sample_rate = torchaudio.load(path)
        waveform = waveform.float()  # Ensure float32
        if sample_rate != self.sampling_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.sampling_rate)
            waveform = resampler(waveform)

        # Apply audio truncation (30 seconds max)
        max_length = 30 * self.sampling_rate  # 30 seconds
        if waveform.shape[-1] > max_length:
            waveform = waveform[..., :max_length]

        return waveform.squeeze()

    def get_audio_embeddings(
        self,
        audio: AudioBatch,
        *,
        task_name: str | None = None,
        prompt_type: PromptType | None = None,
        batch_size: int = 4,
        show_progress_bar: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        all_features = []
        processed_audio = self._process_audio(audio)

        for i in tqdm(
            range(0, len(processed_audio), batch_size),
            desc="Processing audio batches",
            disable=not show_progress_bar,
        ):
            batch = processed_audio[i : i + batch_size]

            # Convert tensors to the format expected by msclap
            # Stack tensors into a batch tensor [batch_size, samples]
            max_length = max(tensor.shape[-1] for tensor in batch)
            batch_tensor = torch.zeros(len(batch), max_length, dtype=torch.float32)

            for idx, tensor in enumerate(batch):
                length = tensor.shape[-1]
                batch_tensor[idx, :length] = tensor

            batch_tensor = batch_tensor.to(self.device)

            try:
                with torch.no_grad():
                    # Use the internal audio encoder directly
                    # [0] gives audio embeddings, [1] gives class probabilities
                    audio_features = self.model.clap.audio_encoder(batch_tensor)[0]

                    # Normalize embeddings
                    audio_features = audio_features / audio_features.norm(
                        dim=-1, keepdim=True
                    )
                    all_features.append(audio_features.cpu().numpy())

            except Exception as e:
                logger.warning(
                    f"⚠️  BATCH processing failed, falling back to individual processing: {e}"
                )
                # Fallback to individual processing
                for tensor in batch:
                    single_tensor = tensor.unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        audio_features = self.model.clap.audio_encoder(single_tensor)[0]
                        audio_features = audio_features / audio_features.norm(
                            dim=-1, keepdim=True
                        )
                        all_features.append(audio_features.cpu().numpy())

        return np.vstack(all_features)

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
