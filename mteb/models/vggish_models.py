from __future__ import annotations

from functools import partial
from typing import Any

import numpy as np
import torch
import torchaudio
from torch.utils.data import DataLoader
from tqdm import tqdm

from mteb.model_meta import ModelMeta
from mteb.requires_package import requires_package


def vggish_loader(**kwargs):
    """Factory function to create a VGGish model wrapper."""
    requires_package(
        vggish_loader,
        "torch_vggish_yamnet",
        "google/vggish",
        "pip install 'mteb[torch-vggish-yamnet]'",
    )
    from torch_vggish_yamnet import vggish
    from torch_vggish_yamnet.input_proc import WaveformToInput

    class VGGishWrapper:
        def __init__(
            self,
            device: str = "cuda" if torch.cuda.is_available() else "cpu",
            **kwargs: Any,
        ):
            self.device = device
            self.model = vggish.get_vggish(with_classifier=False, pretrained=True)
            self.model.eval()
            self.model = self.model.to(self.device)

            self.converter = WaveformToInput()
            self.sampling_rate = 16000  # VGGish requires 16kHz audio
            self.embed_dim = 128  # VGGish embedding dimension
            self.min_audio_length = 0.96
            self.min_samples = int(self.min_audio_length * self.sampling_rate)

        def _process_audio(self, audio):
            processed_audio = []

            if isinstance(audio, DataLoader):
                for batch in audio:
                    processed_audio.extend(self._handle_batch(batch))
            else:
                processed_audio = self._handle_batch(audio)

            return processed_audio

        def _handle_batch(self, batch):
            waveforms = []

            if isinstance(batch, tuple):  # Handle (audio, metadata) tuples
                for audio, _ in batch:
                    print("Audio shape after convert:", audio.shape)
                    waveforms.append(self._convert_audio(audio))
            else:
                for item in batch:
                    if isinstance(item, dict):
                        if "array" in item:
                            audio = item["array"]
                            if isinstance(audio, np.ndarray):
                                audio = torch.from_numpy(audio)
                            audio = audio.float()
                            if item["sampling_rate"] != self.sampling_rate:
                                resampler = torchaudio.transforms.Resample(
                                    item["sampling_rate"], self.sampling_rate
                                )
                                audio = resampler(audio)
                            waveforms.append(self._convert_audio(audio))
                        elif "path" in item:
                            waveforms.append(self._load_audio_file(item["path"]))
                    elif isinstance(
                        item, (np.ndarray, torch.Tensor)
                    ):  ## beijing opera goes here. torch.Tensor.
                        waveforms.append(item)
                    elif isinstance(item, str):
                        waveforms.append(self._load_audio_file(item))

            return waveforms

        def _convert_audio(self, audio):
            if isinstance(audio, np.ndarray):
                audio = torch.from_numpy(audio)
            audio = audio.float().squeeze()

            if audio.ndim > 1:
                audio = audio.mean(dim=0)

            # Normalize to [-1.0, 1.0] if needed
            if audio.abs().max() > 1.0:
                audio = audio / audio.abs().max()

            # Ensure audio is at least 0.96s (15,360 samples at 16kHz)
            min_len = 15360
            if audio.shape[-1] < min_len:
                pad_len = min_len - audio.shape[-1]
                audio = torch.nn.functional.pad(audio, (0, pad_len))

            if audio.shape[-1] < self.min_samples:
                pad_amount = self.min_samples - audio.shape[-1]
                audio = torch.nn.functional.pad(audio, (0, pad_amount))
            return audio

        def _load_audio_file(self, path):
            waveform, sample_rate = torchaudio.load(path)
            waveform = waveform.squeeze().float()
            if sample_rate != self.sampling_rate:
                resampler = torchaudio.transforms.Resample(
                    sample_rate, self.sampling_rate
                )
                waveform = resampler(waveform)
            return self._convert_audio(waveform)

        def get_audio_embeddings(
            self, audio, *, task_name=None, prompt_type=None, batch_size=4, **kwargs
        ):
            processed_audio = self._process_audio(audio)
            all_embeddings = []

            with torch.no_grad():
                for i in tqdm(
                    range(0, len(processed_audio), batch_size), desc="Processing audio"
                ):
                    batch = processed_audio[i : i + batch_size]

                    batch_embeddings = []
                    for audio_data in batch:
                        # Convert to input format expected by VGGish
                        input_tensor = self.converter(
                            audio_data.float(), self.sampling_rate
                        )
                        input_tensor = input_tensor.to(self.device)

                        # Skip empty tensors
                        if input_tensor.numel() == 0:
                            print("Creating zero embedding for empty tensor")
                            zero_embedding = torch.zeros(
                                self.embed_dim, device=self.device
                            )
                            batch_embeddings.append(zero_embedding.cpu().numpy())
                            continue

                        # Ensure correct tensor format for VGGish
                        if input_tensor.dim() == 4:  # [batch, channels, height, width]
                            if input_tensor.shape[1] == 3:
                                # Convert 3-channel to 1-channel by taking the mean
                                input_tensor = input_tensor.mean(dim=1, keepdim=True)
                        elif (
                            input_tensor.dim() == 3
                        ):  # Could be [channels, height, width] or [batch, height, width]
                            if input_tensor.shape[0] == 3:
                                # This is [3, height, width] - treat as 3 channels, add batch dim
                                input_tensor = input_tensor.mean(
                                    dim=0, keepdim=True
                                ).unsqueeze(0)
                            elif input_tensor.shape[0] > 0:
                                # This might be [batch, height, width] - add channel dim
                                input_tensor = input_tensor.unsqueeze(1)

                        # Final check: ensure we have [batch, 1, height, width]
                        if input_tensor.dim() == 3:
                            input_tensor = input_tensor.unsqueeze(1)  # Add channel dim
                        elif input_tensor.dim() == 2:
                            input_tensor = input_tensor.unsqueeze(0).unsqueeze(
                                0
                            )  # Add batch and channel dims

                        embedding = self.model(input_tensor)

                        # Use mean pooling if needed
                        if len(embedding.shape) > 1:
                            embedding = torch.mean(embedding, dim=0)

                        batch_embeddings.append(embedding.cpu().numpy())

                    all_embeddings.extend(batch_embeddings)

            if not all_embeddings:
                return np.zeros((0, self.embed_dim))

            return np.array(all_embeddings)

        def encode(self, inputs, *, task_name, prompt_type=None, **kwargs):
            return self.get_audio_embeddings(
                inputs, task_name=task_name, prompt_type=prompt_type, **kwargs
            )

    return VGGishWrapper(**kwargs)


vggish = ModelMeta(
    loader=partial(vggish_loader),
    name="google/vggish",
    languages=["eng-Latn"],
    open_weights=True,
    revision="1",
    release_date="2019-06-13",
    max_tokens=float("inf"),
    n_parameters=72_141_184,
    memory_usage_mb=275,
    embed_dim=128,
    license="apache-2.0",
    reference="https://github.com/tensorflow/models/tree/master/research/audioset/vggish",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=False,
    public_training_code="https://github.com/tensorflow/models/tree/master/research/audioset/vggish",
    public_training_data="https://research.google.com/audioset/",
    training_datasets={"AudioSet": ["train"]},
    modalities=["audio"],
)
