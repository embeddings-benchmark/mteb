import logging
from collections.abc import Iterable
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, EncodecModel

from mteb.models import ModelMeta
from mteb.models.abs_encoder import AbsEncoder
from mteb.types import Array, PromptType

logger = logging.getLogger(__name__)


class EncodecWrapper(AbsEncoder):
    def __init__(
        self,
        model_name: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_audio_length_seconds: float = 30.0,
        **kwargs: Any,
    ):
        self.model_name = model_name
        self.device = device
        self.max_audio_length_seconds = max_audio_length_seconds

        self.model = EncodecModel.from_pretrained(model_name).to(device)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.sampling_rate = self.processor.sampling_rate  # 24000 Hz typically

    def _process_audio(self, audio) -> list[torch.Tensor]:
        processed_audio = []

        if isinstance(audio, DataLoader):
            for batch in audio:
                processed_audio.extend(self._handle_batch(batch))
        else:
            processed_audio = self._handle_batch(audio)

        return processed_audio

    def _handle_batch(
        self, batch: Array | Iterable[tuple[Array, str]]
    ) -> list[torch.Tensor]:
        import torchaudio

        waveforms = []

        if isinstance(batch, tuple):
            for audio, _ in batch:
                waveforms.append(self._convert_audio(audio))
        else:
            for item in batch:
                if isinstance(item, dict):
                    if "array" in item:
                        audio = item["array"]
                        if isinstance(audio, np.ndarray):
                            audio = torch.from_numpy(audio).float()
                        elif isinstance(audio, list):
                            audio = torch.tensor(audio).float()
                        else:
                            audio = audio.float()
                        # Check for empty audio before resampling
                        if audio.numel() == 0:
                            logger.warning(
                                "Empty audio array from dataset - creating null audio marker"
                            )
                            # Create special marker audio to maintain alignment
                            sr = item.get("sampling_rate", self.sampling_rate)
                            min_samples = max(
                                500, int(0.1 * sr)
                            )  # At least 500 samples or 100ms
                            audio = torch.full(
                                (min_samples,), -999.0, dtype=torch.float32
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

    def _convert_audio(self, audio: Array) -> torch.Tensor:
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio)

        # Ensure float type
        audio = audio.float()

        # Convert to mono if needed (EnCodec can work with stereo, but for embedding we use mono)
        if audio.dim() > 1 and audio.shape[0] > 1:  # If multi-channel
            audio = torch.mean(audio, dim=0, keepdim=True)  # Convert to mono

        audio = audio.squeeze()

        # Handle empty audio by returning a special marker
        if audio.numel() == 0:
            logger.warning(
                "Empty audio tensor encountered - will create null embedding"
            )
            # Return a special tensor that we can identify later
            # Use sufficient samples to avoid processing issues
            min_samples = max(
                500, int(0.1 * self.sampling_rate)
            )  # At least 500 samples or 100ms
            audio = torch.full(
                (min_samples,), -999.0, dtype=torch.float32
            )  # Special marker value

        return audio

    def _load_audio_file(self, path: str) -> torch.Tensor:
        import torchaudio

        try:
            waveform, sample_rate = torchaudio.load(path)
        except Exception as e:
            logger.warning(
                f"Failed to load audio file {path}: {e} - creating null audio marker"
            )
            # Create special marker audio to maintain alignment
            min_samples = max(500, int(0.1 * self.sampling_rate))
            return torch.full((min_samples,), -999.0, dtype=torch.float32)

        # Convert to mono if needed
        if waveform.shape[0] > 1:  # If multi-channel
            waveform = torch.mean(waveform, dim=0, keepdim=True)  # Convert to mono

        if sample_rate != self.sampling_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.sampling_rate)
            waveform = resampler(waveform)

        waveform = waveform.squeeze()

        # Handle empty audio files
        if waveform.numel() == 0:
            logger.warning(f"Empty audio file: {path} - creating null audio marker")
            min_samples = max(500, int(0.1 * self.sampling_rate))
            waveform = torch.full((min_samples,), -999.0, dtype=torch.float32)

        return waveform

    def get_audio_embeddings(
        self,
        audio,
        *,
        task_name: str | None = None,
        prompt_type: PromptType | None = None,
        batch_size: int = 4,
        show_progress_bar: bool = True,
        **kwargs: Any,
    ) -> torch.Tensor:
        processed_audio = self._process_audio(audio)
        all_embeddings = []

        with torch.no_grad():
            for i in tqdm(
                range(0, len(processed_audio), batch_size),
                disable=not show_progress_bar,
            ):
                batch = processed_audio[i : i + batch_size]

                # Process audio through EnCodec's processor
                max_samples = int(self.max_audio_length_seconds * self.sampling_rate)
                batch_np = []
                null_indices = []  # Track which samples are null markers

                for idx, audio in enumerate(batch):
                    audio_np = audio[:max_samples].cpu().numpy()

                    # Check if this is a null marker (all values are -999.0)
                    if len(audio_np) > 0 and np.all(np.abs(audio_np + 999.0) < 1e-6):
                        null_indices.append(idx)
                        # Replace with minimal silence for processing
                        audio_np = (
                            np.zeros_like(audio_np) + 1e-6
                        )  # Very quiet but not zero

                    batch_np.append(audio_np)

                inputs = self.processor(
                    raw_audio=batch_np,
                    sampling_rate=self.sampling_rate,
                    return_tensors="pt",
                    padding="max_length",
                    max_length=max_samples,
                ).to(self.device)

                # Get the latent representations directly from the encoder
                latent = self.model.encoder(inputs.input_values)

                # Apply mean pooling over the time dimension to get fixed-size embeddings
                embeddings = torch.mean(latent, dim=2)  # Average over time dimension

                # Normalize embeddings
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

                # Replace null marker embeddings with special null embeddings
                if null_indices:
                    for idx in null_indices:
                        # Create a null embedding (all zeros)
                        embeddings[idx] = torch.zeros_like(embeddings[idx])
                        logger.debug(
                            f"Created null embedding for empty audio at batch index {idx}"
                        )

                all_embeddings.append(embeddings.cpu())

        if all_embeddings:
            return torch.cat(all_embeddings, dim=0)
        else:
            # Return empty tensor with correct embedding dimension
            return torch.zeros((0, self.model.encoder.hidden_size))

    def encode(
        self,
        inputs,
        *,
        task_name: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        raise ValueError("Encodec models only support audio encoding.")


encodec_24khz = ModelMeta(
    loader=EncodecWrapper,
    name="facebook/encodec_24khz",
    languages=["eng-Latn"],
    open_weights=True,
    revision="c1dbe2ae3f1de713481a3b3e7c47f357092ee040",
    release_date="2022-10-25",
    max_tokens=None,
    n_parameters=23_273_218,
    memory_usage_mb=88,
    embed_dim=128,
    license="cc-by-nc-4.0",
    reference="https://huggingface.co/facebook/encodec_24khz",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=False,
    public_training_code="https://github.com/facebookresearch/encodec",
    public_training_data=None,
    training_datasets=None,  # ["AudioSet", "VCTK", "DNS-Challenge"],
    modalities=["audio"],
)
