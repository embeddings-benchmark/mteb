from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

import numpy as np
import torch
import torchaudio
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import ASTFeatureExtractor, ASTModel

from mteb.models import ModelMeta
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.models_protocols import AudioBatch
from mteb.types import Array, PromptType

logger = logging.getLogger(__name__)


class ASTWrapper(AbsEncoder):
    def __init__(
        self,
        model_name: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs: Any,
    ):
        self.model_name = model_name
        self.device = device

        self.feature_extractor = ASTFeatureExtractor.from_pretrained(model_name)
        self.model = ASTModel.from_pretrained(model_name).to(self.device)
        self.sampling_rate = self.feature_extractor.sampling_rate

    def _process_audio(self, audio: AudioBatch) -> list[torch.Tensor]:
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
        waveforms = []

        if isinstance(batch, tuple):  # Handle (audio, metadata) tuples
            for audio, _ in batch:
                waveforms.append(self._convert_audio_from_numpy(audio))
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
                            # Need at least 400 samples for AST feature extraction
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
                        waveforms.append(self._convert_audio_from_numpy(audio))
                    elif "path" in item:
                        waveforms.append(self._load_audio_file(item["path"]))
                elif isinstance(item, (np.ndarray, torch.Tensor)):
                    waveforms.append(self._convert_audio_from_numpy(item))
                elif isinstance(item, str):
                    waveforms.append(self._load_audio_file(item))

        return waveforms

    def _convert_audio_from_numpy(self, audio: Array) -> torch.Tensor:
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio)
        audio = audio.squeeze()

        # Handle empty audio by returning a special marker
        if audio.numel() == 0:
            logger.warning(
                "Empty audio tensor encountered - will create null embedding"
            )
            # Return a special tensor that we can identify later
            # Need at least 400 samples for AST feature extraction (window size requirement)
            min_samples = max(
                500, int(0.1 * self.sampling_rate)
            )  # At least 500 samples or 100ms
            audio = torch.full(
                (min_samples,), -999.0, dtype=torch.float32
            )  # Special marker value

        return audio

    def _load_audio_file(self, path: str) -> torch.Tensor:
        try:
            waveform, sample_rate = torchaudio.load(path)
        except Exception as e:
            logger.warning(
                f"Failed to load audio file {path}: {e} - creating null audio marker"
            )
            # Create special marker audio to maintain alignment
            # Need at least 400 samples for AST feature extraction
            min_samples = max(500, int(0.1 * self.sampling_rate))
            return torch.full((min_samples,), -999.0, dtype=torch.float32)

        if sample_rate != self.sampling_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.sampling_rate)
            waveform = resampler(waveform)
        waveform = waveform.squeeze()

        # Handle empty audio files
        if waveform.numel() == 0:
            logger.warning(f"Empty audio file: {path} - creating null audio marker")
            # Need at least 400 samples for AST feature extraction
            min_samples = max(500, int(0.1 * self.sampling_rate))
            waveform = torch.full((min_samples,), -999.0, dtype=torch.float32)

        return waveform

    def get_audio_embeddings(
        self,
        audio: AudioBatch,
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

                # AST processes raw waveforms directly through its feature extractor
                batch_inputs = []
                null_indices = []  # Track which samples are null markers

                for idx, audio_tensor in enumerate(batch):
                    audio_np = (
                        audio_tensor.numpy()
                        if isinstance(audio_tensor, torch.Tensor)
                        else audio_tensor
                    )

                    # Check if this is a null marker (all values are -999.0)
                    if len(audio_np) > 0 and np.all(np.abs(audio_np + 999.0) < 1e-6):
                        null_indices.append(idx)
                        # Replace with minimal silence for processing
                        audio_np = (
                            np.zeros_like(audio_np) + 1e-6
                        )  # Very quiet but not zero

                    batch_inputs.append(audio_np)

                inputs = self.feature_extractor(
                    batch_inputs,
                    sampling_rate=self.sampling_rate,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                ).to(self.device)

                outputs = self.model(**inputs)

                # AST's pooled output is the [CLS] token embedding
                embeddings = outputs.pooler_output

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
            return torch.zeros((0, self.model.config.hidden_size))

    def encode(
        self,
        inputs: AudioBatch,
        *,
        task_name: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        raise ValueError("AST models only support audio encoding.")


# Model metadata
ast_audioset = ModelMeta(
    loader=ASTWrapper,
    name="MIT/ast-finetuned-audioset-10-10-0.4593",
    languages=["eng-Latn"],
    open_weights=True,
    revision="f826b80d28226b62986cc218e5cec390b1096902",
    release_date="2021-07-08",
    max_tokens=None,
    n_parameters=86_600_000,
    memory_usage_mb=330,
    embed_dim=768,
    license="apache-2.0",
    reference="https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=False,
    public_training_code="https://github.com/YuanGongND/ast",
    public_training_data="https://research.google.com/audioset/dataset/index.html",
    training_datasets={},  # "AudioSet": ["train"]},
    modalities=["audio"],
)
