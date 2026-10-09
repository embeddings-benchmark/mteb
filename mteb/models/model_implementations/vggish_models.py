from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
from tqdm.auto import tqdm

from mteb.models.abs_encoder import AbsEncoder
from mteb.models.modality_collators import seconds_to_samples
from mteb.models.model_meta import ModelMeta

if TYPE_CHECKING:
    import torch
    from torch.utils.data import DataLoader

    from mteb import TaskMetadata
    from mteb.models.models_protocols import EncoderProtocol
    from mteb.types import Array, BatchedInput, PromptType
    from mteb.types._encoder_io import AudioInput

logger = logging.getLogger(__name__)


def vggish_loader(*args: Any, **kwargs: Any) -> EncoderProtocol:
    """Factory function to create a VGGish model wrapper."""
    import torch
    import torchaudio
    from torch_vggish_yamnet import vggish

    class VGGishWrapper(AbsEncoder):
        def __init__(
            self,
            device: str | None = None,
            # no limit: 0.96 s patches, no max
            # https://github.com/harritaylor/torchvggish/blob/master/torchvggish/vggish_params.py
            max_audio_length_seconds: float | None = None,
            **kwargs: Any,
        ):
            import torch

            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"

            self.device = device
            self.max_audio_length_seconds = max_audio_length_seconds
            self.model = vggish.get_vggish(with_classifier=False, pretrained=True)
            # drop the port's final ReLU: the official embedding layer has no activation
            # https://github.com/tensorflow/models/blob/master/research/audioset/vggish/vggish_slim.py
            self.model.embeddings = self.model.embeddings[:-1]
            self.model.eval().to(self.device)

            self.sampling_rate = 16000
            self.embed_dim = 128
            # the official front end gives no example for clips shorter than one
            # 0.96 s example plus the STFT window minus one hop, so pad to that
            self.min_samples = 15_600  # 0.975 s

        def _resample_audio(
            self, audio: torch.Tensor, source_rate: int
        ) -> torch.Tensor:
            """Resample audio to target sampling rate."""
            if source_rate != self.sampling_rate:
                resampler = torchaudio.transforms.Resample(
                    source_rate, self.sampling_rate
                )
                return resampler(audio)
            return audio

        def _normalize_audio(self, audio: Array) -> torch.Tensor:
            """Normalize and pad audio to minimum length."""
            if isinstance(audio, np.ndarray):
                audio = torch.from_numpy(audio)

            audio = audio.float().squeeze()
            if audio.ndim > 1:
                audio = audio.mean(dim=0)

            # Apply audio truncation
            max_length = seconds_to_samples(
                self.max_audio_length_seconds, self.sampling_rate
            )
            if max_length is not None and audio.shape[-1] > max_length:
                audio = audio[..., :max_length]

            # Normalize to [-1.0, 1.0]
            if audio.numel() > 0 and audio.abs().max() > 1.0:
                audio = audio / audio.abs().max()  # noqa: PLR6104

            # Pad to minimum length
            if audio.shape[-1] < self.min_samples:
                pad_amount = self.min_samples - audio.shape[-1]
                audio = torch.nn.functional.pad(audio, (0, pad_amount))

            return audio

        def _prepare_input_tensor(self, audio_data: Array) -> torch.Tensor:
            """Convert audio to VGGish input format and handle tensor dimensions."""
            if isinstance(audio_data, np.ndarray):
                audio_data = torch.from_numpy(audio_data)

            if audio_data.ndim == 1:
                audio_data = audio_data.unsqueeze(0)

            log_mel = _log_mel(audio_data.squeeze(0).double(), log_offset=0.01)
            # non-overlapping 0.96 s examples: [examples, 1, 96 frames, 64 mels]
            patches = log_mel.float().unfold(0, 96, 96)
            input_tensor = patches.transpose(1, 2).unsqueeze(1).to(self.device)

            return input_tensor

        def get_audio_embeddings(
            self,
            inputs: DataLoader[AudioInput],
            show_progress_bar: bool = True,
            **kwargs: Any,
        ) -> Array:
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
                    sr = a["sampling_rate"]
                    if sr is None:
                        warnings.warn(
                            f"No sampling_rate provided for an audio sample. "
                            f"Assuming {self.sampling_rate} Hz (model default).",
                            stacklevel=2,
                        )
                        sr = self.sampling_rate

                    # Resample if needed
                    audio = self._resample_audio(array.float(), sr)
                    # Normalize
                    audio = self._normalize_audio(audio)

                    with torch.no_grad():
                        input_tensor = self._prepare_input_tensor(audio)
                        embedding = self.model(input_tensor)

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
                raise ValueError("VGGishWrapper only supports audio inputs.")
            return self.get_audio_embeddings(inputs, **kwargs)

    return VGGishWrapper(**kwargs)


def _log_mel(audio: torch.Tensor, log_offset: float) -> torch.Tensor:
    """Log-mel spectrogram [frames, 64 mels] of 16 kHz audio, as the official AudioSet code.

    https://github.com/tensorflow/models/blob/master/research/audioset/vggish/mel_features.py
    """
    import torch

    # 25 ms periodic Hann window, 10 ms hop, no centring, 512-point FFT
    frames = audio.unfold(-1, 400, 160)
    window = torch.hann_window(400, periodic=True, dtype=audio.dtype)
    magnitude = torch.fft.rfft(frames * window, n=512).abs()

    # 64 HTK mel bands between 125 and 7500 Hz, DC bin zeroed
    def hertz_to_mel(hertz: float | np.ndarray) -> np.ndarray:
        return 1127.0 * np.log(1.0 + hertz / 700.0)

    bins_mel = hertz_to_mel(np.linspace(0.0, 8000.0, 257))[:, None]
    edges = np.linspace(hertz_to_mel(125.0), hertz_to_mel(7500.0), 64 + 2)
    lower, center, upper = edges[:-2], edges[1:-1], edges[2:]
    mel_matrix = np.maximum(
        0.0,
        np.minimum(
            (bins_mel - lower) / (center - lower), (upper - bins_mel) / (upper - center)
        ),
    )
    mel_matrix[0, :] = 0.0

    mel = magnitude @ torch.from_numpy(mel_matrix).to(audio.dtype)
    return torch.log(mel + log_offset)


vggish = ModelMeta(
    loader=vggish_loader,
    name="google/vggish",
    languages=["eng-Latn"],
    open_weights=True,
    revision="1",
    release_date="2019-06-13",
    max_tokens=float("inf"),
    n_parameters=72_141_184,
    n_embedding_parameters=0,
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
        "AudioSetMini",
    },
    modalities=["audio"],
    citation="""
@inproceedings{hershey2017cnn,
    author = {Hershey, Shawn and Chaudhuri, Sourish and Ellis, Daniel P. W. and Gemmeke, Jort F. and Jansen, Aren and Moore, R. Channing and Plakal, Manoj and Platt, Devin and Saurous, Rif A. and Seybold, Bryan and Slaney, Malcolm and Weiss, Ron J. and Wilson, Kevin},
    title = {CNN architectures for large-scale audio classification},
    year = {2017},
    publisher = {IEEE Press},
    url = {https://doi.org/10.1109/ICASSP.2017.7952132},
    doi = {10.1109/ICASSP.2017.7952132},
    booktitle = {2017 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
    pages = {131–135},
    numpages = {5},
    location = {New Orleans, LA, USA}
}
""",
    extra_requirements_groups=["torch-vggish-yamnet"],
)
