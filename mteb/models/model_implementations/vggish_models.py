from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from mteb._requires_package import requires_package
from mteb.models import ModelMeta

logger = logging.getLogger(__name__)


def vggish_loader(**kwargs):
    """Factory function to create a VGGish model wrapper."""
    requires_package(
        vggish_loader,
        "torch_vggish_yamnet",
        "google/vggish",
        "pip install 'mteb[torch-vggish-yamnet]'",
    )
    import torchaudio
    from torch_vggish_yamnet import vggish
    from torch_vggish_yamnet.input_proc import WaveformToInput

    class VGGishWrapper:
        def __init__(
            self,
            device: str = "cuda" if torch.cuda.is_available() else "cpu",
            max_audio_length_seconds: float = 30.0,
            **kwargs: Any,
        ):
            self.device = device
            self.max_audio_length_seconds = max_audio_length_seconds
            self.model = vggish.get_vggish(with_classifier=False, pretrained=True)
            self.model.eval().to(self.device)

            self.converter = WaveformToInput()
            self.sampling_rate = 16000
            self.embed_dim = 128
            self.min_samples = int(0.96 * self.sampling_rate)  # 15,360 samples

        def _resample_audio(self, audio, source_rate):
            """Resample audio to target sampling rate."""
            if source_rate != self.sampling_rate:
                resampler = torchaudio.transforms.Resample(
                    source_rate, self.sampling_rate
                )
                return resampler(audio)
            return audio

        def _normalize_audio(self, audio):
            """Normalize and pad audio to minimum length."""
            if isinstance(audio, np.ndarray):
                audio = torch.from_numpy(audio)

            audio = audio.float().squeeze()
            if audio.ndim > 1:
                audio = audio.mean(dim=0)

            # Apply audio truncation
            max_length = int(self.max_audio_length_seconds * self.sampling_rate)
            if audio.shape[-1] > max_length:
                audio = audio[..., :max_length]

            # Normalize to [-1.0, 1.0]
            if audio.abs().max() > 1.0:
                audio = audio / audio.abs().max()

            # Pad to minimum length
            if audio.shape[-1] < self.min_samples:
                pad_amount = self.min_samples - audio.shape[-1]
                audio = torch.nn.functional.pad(audio, (0, pad_amount))

            return audio

        def _process_audio_item(self, item):
            """Process a single audio item into normalized tensor."""
            if isinstance(item, dict):
                if "array" in item:
                    audio = (
                        torch.from_numpy(item["array"])
                        if isinstance(item["array"], np.ndarray)
                        else item["array"]
                    )
                    audio = self._resample_audio(audio.float(), item["sampling_rate"])
                    return self._normalize_audio(audio)
                elif "path" in item:
                    return self._load_audio_file(item["path"])
            elif isinstance(item, (np.ndarray, torch.Tensor)):
                return item
            elif isinstance(item, str):
                return self._load_audio_file(item)

            return item

        def _process_audio(self, audio):
            """Process audio input into list of normalized tensors."""
            if isinstance(audio, DataLoader):
                # Force single-threaded processing to avoid pickling issues
                audio.num_workers = 0

                processed = []
                for batch in audio:
                    processed.extend(self._process_batch(batch))
                return processed
            else:
                return self._process_batch(audio)

        def _process_batch(self, batch):
            """Process a batch of audio items."""
            if isinstance(batch, tuple):  # Handle (audio, metadata) tuples
                waveforms = []
                for audio, _ in batch:
                    waveforms.append(self._normalize_audio(audio))
                return waveforms

            return [self._process_audio_item(item) for item in batch]

        def _load_audio_file(self, path):
            """Load and process audio file."""
            waveform, sample_rate = torchaudio.load(path)
            waveform = self._resample_audio(waveform.squeeze().float(), sample_rate)
            return self._normalize_audio(waveform)

        def _prepare_input_tensor(self, audio_data):
            """Convert audio to VGGish input format and handle tensor dimensions."""
            if isinstance(audio_data, np.ndarray):
                audio_data = torch.from_numpy(audio_data)

            if audio_data.ndim == 1:
                audio_data = audio_data.unsqueeze(0)

            input_tensor = self.converter(audio_data.float(), self.sampling_rate).to(
                self.device
            )

            if input_tensor.numel() == 0:
                return None

            # Handle different tensor dimensions
            if input_tensor.dim() == 4 and input_tensor.shape[1] == 3:
                # [batch, 3, height, width] -> [batch, 1, height, width]
                input_tensor = input_tensor.mean(dim=1, keepdim=True)
            elif input_tensor.dim() == 3:
                if input_tensor.shape[0] == 3:
                    # [3, height, width] -> [1, 1, height, width]
                    input_tensor = input_tensor.mean(dim=0, keepdim=True).unsqueeze(0)
                else:
                    # [batch, height, width] -> [batch, 1, height, width]
                    input_tensor = input_tensor.unsqueeze(1)
            elif input_tensor.dim() == 2:
                # [height, width] -> [1, 1, height, width]
                input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
            elif input_tensor.dim() == 3:
                # [batch, height, width] -> [batch, 1, height, width]
                input_tensor = input_tensor.unsqueeze(1)

            return input_tensor

        def get_audio_embeddings(
            self,
            audio,
            *,
            task_name=None,
            prompt_type=None,
            batch_size=4,
            show_progress_bar=True,
            **kwargs,
        ):
            """Generate embeddings for audio inputs."""
            processed_audio = self._process_audio(audio)
            all_embeddings = []

            with torch.no_grad():
                for i in tqdm(
                    range(0, len(processed_audio), batch_size),
                    desc="Processing audio",
                    disable=not show_progress_bar,
                ):
                    batch = processed_audio[i : i + batch_size]
                    batch_embeddings = []

                    for audio_data in batch:
                        input_tensor = self._prepare_input_tensor(audio_data)

                        if input_tensor is None:
                            # Create zero embedding for empty tensor
                            logger.debug("Creating zero embedding for empty tensor")
                            zero_embedding = torch.zeros(
                                self.embed_dim, device=self.device
                            )
                            batch_embeddings.append(zero_embedding.cpu().numpy())
                            continue

                        embedding = self.model(input_tensor)

                        # Use mean pooling if needed
                        if len(embedding.shape) > 1:
                            embedding = torch.mean(embedding, dim=0)

                        batch_embeddings.append(embedding.cpu().numpy())

                    all_embeddings.extend(batch_embeddings)

            return (
                np.array(all_embeddings)
                if all_embeddings
                else np.zeros((0, self.embed_dim))
            )

        def encode(self, inputs, *, task_name, prompt_type=None, **kwargs):
            raise ValueError("vggish models only support audio encoding.")

    return VGGishWrapper(**kwargs)


vggish = ModelMeta(
    loader=vggish_loader,
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
    training_datasets={
        "AudioSet",
    },
    modalities=["audio"],
)
