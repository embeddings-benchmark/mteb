from __future__ import annotations

from functools import partial
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import ClapProcessor, ClapModel

from mteb.encoder_interface import AudioBatch, AudioData, PromptType
from mteb.model_meta import ModelMeta

class ClapZeroShotWrapper:
    def __init__(
        self,
        model_name: str = "laion/clap-htsat-unfused",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs: Any,
    ):
        self.model_name = model_name
        self.device = device
        self.model = ClapModel.from_pretrained(model_name).to(self.device)
        self.processor = ClapProcessor.from_pretrained(model_name)
        self.sampling_rate = self.processor.feature_extractor.sampling_rate

    def _process_audio(self, audio: AudioBatch) -> list[torch.Tensor]:
        processed_audio = []

        if isinstance(audio, DataLoader):
            for batch in audio:
                processed_audio.extend(self._handle_batch(batch))
        else:
            processed_audio = self._handle_batch(audio)

        return processed_audio

    def _handle_batch(self, batch: AudioData | Iterable[tuple[AudioData, str]]) -> list[torch.Tensor]:
        # Similar to Wav2Vec2AudioWrapper's _handle_batch
        # Process audio data into format expected by CLAP
        pass

    def get_audio_embeddings(
        self,
        audio: AudioBatch,
        **kwargs: Any,
    ) -> np.ndarray:
        processed_audio = self._process_audio(audio)
        inputs = self.processor(audios=processed_audio, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            audio_features = self.model.get_audio_features(**inputs)
            # Normalize embeddings
            audio_features = audio_features / audio_features.norm(dim=-1, keepdim=True)
        
        return audio_features.cpu().numpy()

    def get_text_embeddings(
        self,
        texts: list[str],
        **kwargs: Any,
    ) -> np.ndarray:
        inputs = self.processor(text=texts, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
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

# Model metadata
clap_htsat_unfused = ModelMeta(
    loader=partial(ClapZeroShotWrapper, model_name="laion/clap-htsat-unfused"),
    name="laion/clap-htsat-unfused",
    languages=["en"],
    revision="main",
    release_date="2023-05-22",
    modalities=["audio", "text"],
    n_parameters=None,  # Fill in actual number
    memory_usage_mb=None,  # Fill in actual number
    max_tokens=float("inf"),
    embed_dim=512,  # Verify this
    license="MIT",
    open_weights=True,
    public_training_code="https://github.com/LAION-AI/CLAP",
    public_training_data="LAION-Audio-630K",
    framework=["PyTorch"],
    reference="https://huggingface.co/laion/clap-htsat-unfused",
    similarity_fn_name="cosine",
    use_instructions=False,
    training_datasets=["LAION-Audio-630K"],
)