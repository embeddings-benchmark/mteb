import logging
import warnings
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from mteb import TaskMetadata
from mteb._requires_package import requires_audio_dependencies, requires_package
from mteb.models import ModelMeta
from mteb.types import Array, BatchedInput, PromptType
from mteb.types._encoder_io import AudioInput

logger = logging.getLogger(__name__)


def yamnet_loader(*args, **kwargs):
    """Factory function to create a YAMNet model wrapper."""
    requires_package(
        yamnet_loader,
        "torch_vggish_yamnet",
        "google/yamnet",
        "pip install 'mteb[torch-vggish-yamnet]'",
    )
    from torch_vggish_yamnet import yamnet
    from torch_vggish_yamnet.input_proc import WaveformToInput

    class YAMNetWrapper:
        def __init__(
            self,
            device: str = "cuda" if torch.cuda.is_available() else "cpu",
            max_audio_length_seconds: float = 30.0,
            **kwargs: Any,
        ):
            requires_audio_dependencies()
            self.device = device
            self.max_audio_length_seconds = max_audio_length_seconds
            self.model = yamnet.yamnet(pretrained=True)
            self.model.eval().to(self.device)

            self.converter = WaveformToInput()
            self.sampling_rate = 16000  # YAMNet requires 16kHz audio
            self.embed_dim = 1024  # YAMNet embedding dimension
            self.min_samples = int(0.96 * self.sampling_rate)  # 15,360 samples

        def _resample_audio(self, audio, source_rate):
            """Resample audio to target sampling rate."""
            import torchaudio

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

        def _prepare_input_tensor(self, audio_data):
            """Convert audio to YAMNet input format and handle tensor dimensions."""
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
            if input_tensor.dim() == 4 and input_tensor.shape[0] == 2:
                # [2, batch, height, width] -> [1, batch, height, width]
                input_tensor = input_tensor.mean(dim=0, keepdim=True)
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

            return input_tensor

        def get_audio_embeddings(
            self,
            inputs: DataLoader[AudioInput],
            show_progress_bar=True,
            **kwargs,
        ):
            """Generate embeddings for audio inputs."""
            all_embeddings = []

            for batch in tqdm(
                inputs,
                desc="Processing audio",
                disable=not show_progress_bar,
            ):
                batch_embeddings = []

                for a in batch["audio"]:
                    array = torch.tensor(a["array"], dtype=torch.float32)
                    sr = a.get("sampling_rate", None)
                    if sr is None:
                        warnings.warn(
                            f"No sampling_rate provided for an audio sample. "
                            f"Assuming {self.sampling_rate} Hz (model default)."
                        )
                        sr = self.sampling_rate

                    # Resample if needed
                    audio = self._resample_audio(array.float(), sr)
                    # Normalize
                    audio = self._normalize_audio(audio)

                    with torch.no_grad():
                        input_tensor = self._prepare_input_tensor(audio)

                        if input_tensor is None:
                            # Create zero embedding for empty tensor
                            logger.debug("Creating zero embedding for empty tensor")
                            zero_embedding = torch.zeros(
                                self.embed_dim, device=self.device
                            )
                            batch_embeddings.append(
                                zero_embedding.cpu().detach().numpy()
                            )
                            continue

                        # discard logits
                        embedding, _ = self.model(input_tensor)

                        # Handle 4D tensors with shape [batch, embed_dim, 1, 1]
                        if (
                            embedding.dim() == 4
                            and embedding.shape[2] == 1
                            and embedding.shape[3] == 1
                        ):
                            # [batch, embed_dim, 1, 1] -> [batch, embed_dim]
                            embedding = embedding.squeeze(2).squeeze(2)

                        # Use mean pooling if needed
                        if len(embedding.shape) > 1:
                            embedding = torch.mean(embedding, dim=0)

                        batch_embeddings.append(embedding.cpu().detach().numpy())

                all_embeddings.extend(batch_embeddings)

            return np.array(all_embeddings)

        def encode(
            self,
            inputs: DataLoader[BatchedInput],
            *,
            task_metadata: TaskMetadata,
            hf_split: str,
            hf_subset: str,
            prompt_type: PromptType | None = None,
            **kwargs: Any,
        ) -> Array:
            if "audio" not in inputs.dataset.features:
                raise ValueError("YAMNetWrapper only supports audio inputs.")
            return self.get_audio_embeddings(inputs, **kwargs)

    return YAMNetWrapper(**kwargs)


yamnet = ModelMeta(
    loader=yamnet_loader,
    name="google/yamnet",
    languages=["eng-Latn"],
    open_weights=True,
    revision="1",
    release_date="2020-10-06",
    max_tokens=float("inf"),
    n_parameters=3_751_369,
    memory_usage_mb=14,
    embed_dim=1024,
    license="apache-2.0",
    reference="https://tfhub.dev/google/yamnet/1",
    similarity_fn_name="cosine",
    framework=["TensorFlow"],
    use_instructions=False,
    public_training_code="https://github.com/tensorflow/models/tree/master/research/audioset/yamnet",
    public_training_data="https://research.google.com/audioset/",
    training_datasets={
        "AudioSet",
    },
    modalities=["audio"],
)
