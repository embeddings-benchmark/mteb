from __future__ import annotations

from collections.abc import Iterable
from functools import partial
from typing import Any

import numpy as np
import torch
import torchaudio
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC

from mteb.encoder_interface import AudioBatch, AudioData, PromptType
from mteb.model_meta import ModelMeta


class Wav2Vec2AudioWrapper:
    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-xls-r-300m",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs: Any,
    ):
        self.model_name = model_name
        self.device = device
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name).to(self.device)
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
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
                waveforms.append(self._convert_audio(audio))
        else:
            for item in batch:
                if isinstance(item, (np.ndarray, torch.Tensor)):
                    waveforms.append(self._convert_audio(item))
                elif isinstance(item, str):
                    waveforms.append(self._load_audio_file(item))

        return waveforms

    def _convert_audio_from_numpy(self, audio: AudioData) -> torch.Tensor:
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio)
        return audio.squeeze()

    def _load_audio_file(self, path: str) -> torch.Tensor:
        waveform, sample_rate = torchaudio.load(path)
        if sample_rate != self.sampling_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.sampling_rate)
            waveform = resampler(waveform)
        return waveform.squeeze()

    def get_audio_embeddings(
        self,
        audio: AudioBatch,
        *,
        task_name: str | None = None,
        prompt_type: PromptType | None = None,
        batch_size: int = 4,
        **kwargs: Any,
    ) -> torch.Tensor:
        processed_audio = self._process_audio(audio)
        all_embeddings = []

        with torch.no_grad():
            for i in tqdm(range(0, len(processed_audio), batch_size)):
                batch = processed_audio[i : i + batch_size]
                inputs = self.feature_extractor(
                    batch,
                    sampling_rate=self.sampling_rate,
                    return_tensors="pt",
                    padding=True,
                    return_attention_mask=True,
                ).to(self.device)

                outputs = self.model(
                    inputs.input_values,
                    attention_mask=inputs.attention_mask,
                    output_hidden_states=True,
                )

                last_hidden_state = outputs.hidden_states[-1]
                embeddings = torch.mean(last_hidden_state, dim=1)
                all_embeddings.append(embeddings.cpu())

        return torch.cat(all_embeddings, dim=0)

    def encode(
        self,
        inputs: AudioBatch,
        *,
        task_name: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        return self.get_audio_embeddings(inputs, task_name=task_name, **kwargs).numpy()


# VERIFY THE INFO BELOW, i got r1 to write this out as a placeholder,,,,,
wav2vec2_xlsr_300m = ModelMeta(
    loader=partial(Wav2Vec2AudioWrapper, model_name="facebook/wav2vec2-xls-r-300m"),
    name="facebook/wav2vec2-xls-r-300m",
    languages=["multilingual"],
    revision="1a640f32ac3e39899438a2931f9924c02f080a54",
    release_date="2021-10-13",
    modalities=["audio"],
    n_parameters=300_000_000,
    memory_usage_mb=1200,
    max_tokens=float("inf"),
    embed_dim=1024,
    license="Apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/pytorch/fairseq",
    public_training_data=None,
    framework=["PyTorch"],
    reference="https://huggingface.co/facebook/wav2vec2-xls-r-300m",
    similarity_fn_name="cosine",
    use_instructions=False,
    training_datasets={},
)

wav2vec2_xlsr_300m_phoneme = ModelMeta(
    loader=partial(
        Wav2Vec2AudioWrapper, model_name="vitouphy/wav2vec2-xls-r-300m-phoneme"
    ),
    name="vitouphy/wav2vec2-xls-r-300m-phoneme",
    languages=["en"],
    revision="bf9913bf096d133cf4eca64ed75981ebf0545c9d",
    release_date="2022-05-19",
    modalities=["audio"],
    n_parameters=300_000_000,
    memory_usage_mb=1200,
    max_tokens=float("inf"),
    embed_dim=1024,
    license="Apache-2.0",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["PyTorch"],
    reference="https://huggingface.co/vitouphy/wav2vec2-xls-r-300m-phoneme",
    similarity_fn_name="cosine",
    use_instructions=False,
    training_datasets=None,
)
wav2vec2_xlsr_1b = ModelMeta(
    loader=partial(Wav2Vec2AudioWrapper, model_name="facebook/wav2vec2-xls-r-1b"),
    name="facebook/wav2vec2-xls-r-1b",
    languages=["multilingual"],
    revision="35eaea9a0ed0f97592277d79208e40ab8917d1e3",
    release_date="2024-09-10",
    modalities=["audio"],
    n_parameters=1_000_000_000,
    memory_usage_mb=4500,
    max_tokens=float("inf"),
    embed_dim=1024,
    license="Apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/pytorch/fairseq",
    public_training_data=None,
    framework=["PyTorch"],
    reference="https://huggingface.co/facebook/wav2vec2-xls-r-1b",
    similarity_fn_name="cosine",
    use_instructions=False,
    training_datasets=None,
)

wav2vec2_xlsr_2b = ModelMeta(
    loader=partial(Wav2Vec2AudioWrapper, model_name="facebook/wav2vec2-xls-r-2b"),
    name="facebook/wav2vec2-xls-r-2b",
    languages=["multilingual"],
    revision="3b6d89d0fabead7da552eaaa07549c7c9c36d303",
    release_date="2024-09-10",
    modalities=["audio"],
    n_parameters=2_000_000_000,
    memory_usage_mb=9000,
    max_tokens=float("inf"),
    embed_dim=1024,
    license="Apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/pytorch/fairseq",
    public_training_data=None,
    framework=["PyTorch"],
    reference="https://huggingface.co/facebook/wav2vec2-xls-r-2b",
    similarity_fn_name="cosine",
    use_instructions=False,
    training_datasets=None,
)

wav2vec2_xlsr_2b_translation = ModelMeta(
    loader=partial(
        Wav2Vec2AudioWrapper, model_name="facebook/wav2vec2-xls-r-2b-21-to-en"
    ),
    name="facebook/wav2vec2-xls-r-2b-21-to-en",
    languages=["multilingual"],
    revision="70239d15f5b39ecbc936a5e214bf401b7f17e210",
    release_date="2024-09-10",
    modalities=["audio"],
    n_parameters=2_000_000_000,
    memory_usage_mb=9200,
    max_tokens=float("inf"),
    embed_dim=1024,
    license="Apache-2.0",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["PyTorch"],
    reference="https://huggingface.co/facebook/wav2vec2-xls-r-2b-21-to-en",
    similarity_fn_name="cosine",
    use_instructions=False,
    training_datasets=None,
)
