from __future__ import annotations

from functools import partial

import numpy as np
import torch
from datasets import Audio
from transformers import (
    AutoProcessor,
    Qwen2AudioForConditionalGeneration,
)

from mteb.encoder_interface import AudioEncoder, PromptType
from mteb.model_meta import ModelMeta


class Qwen2AudioWrapper(AudioEncoder):
    def __init__(self, model_name: str, device: str | None = None, **kwargs):
        super().__init__(device=device, **kwargs)
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B")
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-Audio-7B"
        )

        self.audio_encoder = self.model.audio_tower

        if hasattr(self.model.config.audio_config, "d_model"):
            self.embed_dim = self.model.config.audio_config.d_model
        elif hasattr(self.model.config.audio_config, "hidden_size"):
            self.embed_dim = self.model.config.audio_config.hidden_size
        else:
            self.embed_dim = None
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.audio_encoder = self.audio_encoder.to(self.device)
        print("Qwen2-Audio initialized. Hiden dim:", self.embed_dim)

    def get_audio_embeddings(
        self, audio_files: list[Audio] | Audio, batch_size: int = 32, **kwargs
    ) -> np.ndarray:
        if not isinstance(audio_files, list):
            audio_files = [audio_files]
        all_embeds = []
        for i in range(0, len(audio_files), batch_size):
            batch = audio_files[i : i + batch_size]
            audios = [file["array"] for file in batch]
            sr = batch[0]["sampling_rate"]

            prompt = " ".join(["<|AUDIO|>"] * len(batch))
            inputs = self.processor(
                text=prompt,
                audios=audios,
                sampling_rate=sr,
                return_tensors="pt",
                padding=True,
            )

            input_features = inputs.input_features.to(self.device)
            with torch.no_grad():
                outputs = self.audio_encoder(input_features=input_features)

            embeds = outputs.last_hidden_state.mean(dim=1)
            print(embeds.shape)
            all_embeds.append(embeds.cpu().numpy())

        return np.vstack(all_embeds)

    def encode(
        self,
        audio_files: list[Audio],
        *,
        task_name: str,
        prompt_type: PromptType | None = None,
        **kwargs,
    ) -> np.ndarray:
        return self.get_audio_embeddings(audio_files, **kwargs)


qwen2_audio_meta = ModelMeta(
    loader=partial(Qwen2AudioWrapper, model_name="Qwen/Qwen2-Audio-7B"),
    name="Qwen/Qwen2-Audio-7B",
    languages=["multilingual"],
    open_weights=True,
    revision=None,
    release_date="2024-08-09",
    max_tokens=float("inf"),
    n_parameters=7_000_000_000,
    memory_usage_mb=None,
    embed_dim=1280,
    license="Unknown",
    reference="https://huggingface.co/Qwen/Qwen2-Audio-7B",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
    modalities=["audio"],
)
