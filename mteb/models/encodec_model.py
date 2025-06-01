from __future__ import annotations

from collections.abc import Iterable
from functools import partial
from typing import Any

import numpy as np
import torch
import torchaudio
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import EncodecModel, AutoProcessor

from mteb.encoder_interface import AudioBatch, AudioData, PromptType
from mteb.model_meta import ModelMeta
from mteb.models.wrapper import Wrapper


class EncodecWrapper(Wrapper):
    def __init__(
        self,
        model_name: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs: Any,
    ):
        self.model_name = model_name
        self.device = device

        self.model = EncodecModel.from_pretrained(model_name).to(device)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.sampling_rate = self.processor.sampling_rate  # 24000 Hz typically

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

        if isinstance(batch, tuple):
            for audio, _ in batch:
                waveforms.append(self._convert_audio(audio))
        else:
            for item in batch:
                if isinstance(item, dict):
                    if "array" in item:
                        audio = item["array"]
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

        # Ensure float type
        audio = audio.float()

        # Convert to mono if needed (EnCodec can work with stereo, but for embedding we use mono)
        if audio.dim() > 1 and audio.shape[0] > 1:  # If multi-channel
            audio = torch.mean(audio, dim=0, keepdim=True)  # Convert to mono

        return audio.squeeze()

    def _load_audio_file(self, path: str) -> torch.Tensor:
        waveform, sample_rate = torchaudio.load(path)

        # Convert to mono if needed
        if waveform.shape[0] > 1:  # If multi-channel
            waveform = torch.mean(waveform, dim=0, keepdim=True)  # Convert to mono

        if sample_rate != self.sampling_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.sampling_rate)
            waveform = resampler(waveform)
        return waveform.squeeze()

    def get_audio_embeddings(
        self,
        audio: AudioBatch,
        *,
        task_name: str | None = None,
        prompt_type: PromptType | None = None,
        batch_size: int = 4,
        **kwargs: Any,
    ) -> torch.Tensor:
        processed_audio = self._process_audio(audio)
        all_embeddings = []

        with torch.no_grad():
            for i in tqdm(range(0, len(processed_audio), batch_size)):
                batch = processed_audio[i : i + batch_size]
                
                # Process audio through EnCodec's processor
                inputs = self.processor(
                    raw_audio=[audio.cpu().numpy() for audio in batch],
                    sampling_rate=self.sampling_rate,
                    return_tensors="pt",
                    padding=True,
                ).to(self.device)
                
                # Get the latent representations directly from the encoder
                # This gives continuous embeddings instead of discrete codes
                latent = self.model.encoder(inputs.input_values)
                
                # Apply mean pooling over the time dimension to get fixed-size embeddings
                embeddings = torch.mean(latent, dim=2)  # Average over time dimension
                
                # Normalize embeddings
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
                
                all_embeddings.append(embeddings.cpu())

        if all_embeddings:
            return torch.cat(all_embeddings, dim=0)
        else:
            # Return empty tensor with correct embedding dimension
            return torch.zeros((0, self.model.encoder.hidden_size))

    def encode(
        self,
        inputs: AudioBatch,
        *,
        task_name: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        return self.get_audio_embeddings(inputs, task_name=task_name, **kwargs).numpy()


encodec_24khz = ModelMeta(
    loader=partial(
        EncodecWrapper,
        model_name="facebook/encodec_24khz",
    ),
    name="facebook/encodec_24khz",
    languages=["eng-Latn"], 
    open_weights=True,
    revision="c1dbe2ae3f1de713481a3b3e7c47f357092ee040",
    release_date="2022-10-25",
    max_tokens=None,
    n_parameters=26_900_000,  # ~27M parameters
    memory_usage_mb=110,  # Relatively lightweight
    embed_dim=1024,  # Codebook size
    license="cc-by-nc-4.0",
    reference="https://huggingface.co/facebook/encodec_24khz",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=False,
    public_training_code="https://github.com/facebookresearch/encodec",
    public_training_data=None,
    training_datasets=None, # ["AudioSet", "VCTK", "DNS-Challenge"],
    modalities=["audio"],
)
