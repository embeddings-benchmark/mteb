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
from transformers import ASTFeatureExtractor, ASTModel

from mteb.encoder_interface import AudioBatch, AudioData, PromptType
from mteb.model_meta import ModelMeta
from mteb.models.wrapper import Wrapper

logger = logging.getLogger(__name__)


class ASTWrapper(Wrapper):
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
        self, batch: AudioData | Iterable[tuple[AudioData, str]]
    ) -> list[torch.Tensor]:
        waveforms = []

        if isinstance(batch, tuple):  # Handle (audio, metadata) tuples
            for audio, _ in batch:
                converted = self._convert_audio_from_numpy(audio)
                if converted is not None:
                    waveforms.append(converted)
        else:
            for item in batch:
                converted = None
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
                        converted = self._convert_audio_from_numpy(audio)
                    elif "path" in item:
                        converted = self._load_audio_file(item["path"])
                elif isinstance(item, (np.ndarray, torch.Tensor)):
                    converted = self._convert_audio_from_numpy(item)
                elif isinstance(item, str):
                    converted = self._load_audio_file(item)
                
                if converted is not None:
                    waveforms.append(converted)

        return waveforms

    def _convert_audio_from_numpy(self, audio: AudioData) -> torch.Tensor | None:
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio)
        audio = audio.squeeze()
        
        # Validate audio is not empty
        if audio.numel() == 0:
            logger.warning("Skipping empty audio tensor during processing")
            return None
            
        return audio

    def _load_audio_file(self, path: str) -> torch.Tensor | None:
        try:
            waveform, sample_rate = torchaudio.load(path)
        except Exception as e:
            logger.warning(f"Failed to load audio file {path}: {e}")
            return None
            
        if sample_rate != self.sampling_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.sampling_rate)
            waveform = resampler(waveform)
        waveform = waveform.squeeze()
        
        # Validate audio is not empty
        if waveform.numel() == 0:
            logger.warning(f"Skipping empty audio file: {path}")
            return None
            
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
                for audio_tensor in batch:
                    # Skip empty audio tensors (already logged in _convert_audio_from_numpy)
                    if audio_tensor.numel() == 0:
                        continue
                    
                    audio_np = (
                        audio_tensor.numpy()
                        if isinstance(audio_tensor, torch.Tensor)
                        else audio_tensor
                    )
                    batch_inputs.append(audio_np)

                # Skip batch if no valid audio samples
                if not batch_inputs:
                    logger.warning("Skipping batch with no valid audio samples")
                    continue

                inputs = self.feature_extractor(
                    batch_inputs,
                    sampling_rate=self.sampling_rate,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                ).to(self.device)

                outputs = self.model(**inputs)

                # AST's pooled output is the [CLS] token embedding
                # This is different from mean pooling used in other models
                embeddings = outputs.pooler_output
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
    loader=partial(
        ASTWrapper,
        model_name="MIT/ast-finetuned-audioset-10-10-0.4593",
    ),
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
