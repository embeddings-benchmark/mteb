from __future__ import annotations

from collections.abc import Iterable
from functools import partial
from typing import Any

import numpy as np
import torch
import torchaudio
from torch.utils.data import DataLoader
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

from mteb.encoder_interface import AudioBatch, AudioData, PromptType
from mteb.model_meta import ModelMeta
from mteb.models.wrapper import Wrapper


class Qwen2AudioWrapper(Wrapper):
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-Audio-7B",
        device: str | None = None,
        **kwargs: Any,
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(model_name)

        self.audio_encoder = self.model.audio_tower
        self.model = self.model.to(self.device)
        self.audio_encoder = self.audio_encoder.to(self.device)

        cfg = self.model.config.audio_config
        self.embed_dim = getattr(cfg, "d_model", getattr(cfg, "hidden_size", None))
        self.sampling_rate = 16000

    def _process_audio(self, audio: AudioBatch) -> list[torch.Tensor]:
        processed: list[torch.Tensor] = []
        if isinstance(audio, DataLoader):
            for batch in audio:
                processed.extend(self._handle_batch(batch))
        else:
            processed = self._handle_batch(audio)
        return processed

    def _handle_batch(
        self, batch: AudioData | Iterable[tuple[AudioData, str]]
    ) -> list[torch.Tensor]:
        waveforms: list[torch.Tensor] = []
        if isinstance(batch, tuple):
            for audio, _ in batch:
                waveforms.append(self._convert_audio_from_numpy(audio))
        else:
            for item in batch:
                if isinstance(item, dict):
                    if "array" in item:
                        arr = item["array"]
                        sr = item.get("sampling_rate", None)
                        tensor = (
                            torch.from_numpy(arr).float()
                            if isinstance(arr, np.ndarray)
                            else arr.float()
                        )
                        if sr and sr != self.sampling_rate:
                            resampler = torchaudio.transforms.Resample(
                                sr, self.sampling_rate
                            )
                            tensor = resampler(tensor)
                        waveforms.append(self._convert_audio_from_numpy(tensor))
                    elif "path" in item:
                        waveforms.append(self._load_audio_file(item["path"]))
                elif isinstance(item, (np.ndarray, torch.Tensor)):
                    waveforms.append(self._convert_audio_from_numpy(item))
                elif isinstance(item, str):
                    waveforms.append(self._load_audio_file(item))
        return waveforms

    def _convert_audio_from_numpy(self, audio: AudioData) -> torch.Tensor:
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio)
        if audio.ndim == 2:
            audio = audio.mean(dim=0)
        return audio.squeeze()

    def _load_audio_file(self, path: str) -> torch.Tensor:
        waveform, sr = torchaudio.load(path)
        if waveform.ndim == 2:
            waveform = waveform.mean(dim=0)
        if sr != self.sampling_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sampling_rate)
            waveform = resampler(waveform)
        return waveform.squeeze()

    def _pad_audio_batch(self, batch: list[torch.Tensor]) -> torch.Tensor:
        max_len = max(w.shape[0] for w in batch)
        padded = [torch.nn.functional.pad(w, (0, max_len - w.shape[0])) for w in batch]
        return torch.stack(padded)

    def get_audio_embeddings(
        self,
        audio: AudioBatch,
        *,
        task_name: str | None = None,
        prompt_type: PromptType | None = None,
        batch_size: int = 4,
        **kwargs: Any,
    ) -> torch.Tensor:
        processed = self._process_audio(audio)
        embeddings_list: list[torch.Tensor] = []

        with torch.no_grad():
            for i in range(0, len(processed), batch_size):
                batch = processed[i : i + batch_size]

                audio_list = [w.numpy() for w in batch]
                prompt = " ".join(["<|AUDIO|>"] * len(audio_list))

                inputs = self.processor(
                    text=prompt,
                    audios=audio_list,
                    sampling_rate=self.processor.feature_extractor.sampling_rate,
                    return_tensors="pt",
                    padding=True,
                )

                input_features = inputs.input_features.to(self.device)

                outputs = self.audio_encoder(
                    input_features=input_features,
                    output_hidden_states=True,
                )

                last_hidden = outputs.hidden_states[-1]
                emb = last_hidden.mean(dim=1).cpu()
                embeddings_list.append(emb)

        return torch.cat(embeddings_list, dim=0)

    def encode(
        self,
        inputs: AudioBatch,
        *,
        task_name: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        return self.get_audio_embeddings(inputs, **kwargs).numpy()


qwen2_audio_meta = ModelMeta(
    loader=partial(
        Qwen2AudioWrapper,
        model_name="Qwen/Qwen2-Audio-7B",
        model_revision="dd84470756e6277a71d4d7188773a43cde92696e",
    ),
    name="Qwen/Qwen2-Audio-7B",
    languages=["eng-Latn"],
    open_weights=True,
    revision=None,
    release_date="2024-08-09",
    max_tokens=float("inf"),
    n_parameters=7_000_000_000,
    memory_usage_mb=None,
    embed_dim=1280,
    license="mit",
    reference="https://huggingface.co/Qwen/Qwen2-Audio-7B",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
    modalities=["audio"],
)
