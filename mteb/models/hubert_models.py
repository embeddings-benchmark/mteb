from __future__ import annotations

from collections.abc import Iterable
from functools import partial
from typing import Any

import numpy as np
import torch
import torchaudio
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import HubertModel, Wav2Vec2FeatureExtractor

from mteb.encoder_interface import AudioBatch, AudioData, PromptType
from mteb.model_meta import ModelMeta
from mteb.models.wrapper import Wrapper


class HubertWrapper(Wrapper):
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

        # HuBERT uses the same feature extractor as Wav2Vec2
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = HubertModel.from_pretrained(model_name).to(self.device)
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
                waveforms.append(self._convert_audio_from_numpy(audio))
        else:
            for item in batch:
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
                        waveforms.append(self._convert_audio_from_numpy(audio))
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
        return audio.squeeze()

    def _load_audio_file(self, path: str) -> torch.Tensor:
        waveform, sample_rate = torchaudio.load(path)
        if sample_rate != self.sampling_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.sampling_rate)
            waveform = resampler(waveform)
        return waveform.squeeze()

    def _pad_audio_batch(self, batch):
        batch = [x.reshape(-1) if x.ndim == 0 else x for x in batch]
        max_length = max(audio.shape[0] for audio in batch)
        padded_batch = [
            torch.nn.functional.pad(audio, (0, max_length - audio.shape[0]))
            for audio in batch
        ]
        return torch.stack(padded_batch)

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

                # Pre-process like Wav2Vec2
                batch_tensor = self._pad_audio_batch(batch)

                if batch_tensor.ndim == 1:
                    batch_tensor = batch_tensor.unsqueeze(0)
                elif batch_tensor.ndim > 2:
                    batch_tensor = batch_tensor.view(batch_tensor.size(0), -1)

                inputs = self.feature_extractor(
                    batch_tensor.cpu().numpy(),
                    sampling_rate=self.sampling_rate,
                    return_tensors="pt",
                    padding="longest",
                    truncation=True,
                    max_length=int(self.max_audio_length_seconds * self.sampling_rate),
                    return_attention_mask=True,
                ).to(self.device)

                outputs = self.model(
                    inputs.input_values,
                    attention_mask=inputs.attention_mask,
                    output_hidden_states=True,
                )

                last_hidden_state = outputs.last_hidden_state
                embeddings = torch.mean(last_hidden_state, dim=1)
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
        raise ValueError("Hubert models only support audio encoding.")


# Base model
hubert_base = ModelMeta(
    loader=partial(
        HubertWrapper,
        model_name="facebook/hubert-base-ls960",
    ),
    name="facebook/hubert-base-ls960",
    languages=["eng-Latn"],
    open_weights=True,
    revision="dba3bb02fda4248b6e082697eee756de8fe8aa8a",
    release_date="2021-06-14",  # Paper release date
    max_tokens=float("inf"),
    n_parameters=95_000_000,
    memory_usage_mb=360,
    embed_dim=768,
    license="mit",
    reference="https://huggingface.co/facebook/hubert-base-ls960",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=False,
    public_training_code="https://github.com/pytorch/fairseq/tree/master/examples/hubert",
    public_training_data="https://www.openslr.org/12",  # Link to LibriSpeech Dataset
    training_datasets={},  # "LibriSpeech": ["train"]},
    modalities=["audio"],
)

# Fine-tuned large model
hubert_large_ft = ModelMeta(
    loader=partial(
        HubertWrapper,
        model_name="facebook/hubert-large-ls960-ft",
    ),
    name="facebook/hubert-large-ls960-ft",
    languages=["eng-Latn"],
    open_weights=True,
    revision="ece5fabbf034c1073acae96d5401b25be96709d8",
    release_date="2021-06-14",  # Paper release date
    max_tokens=float("inf"),
    n_parameters=317_000_000,
    memory_usage_mb=1203,
    embed_dim=1024,
    license="mit",
    reference="https://huggingface.co/facebook/hubert-large-ls960-ft",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=False,
    public_training_code="https://github.com/pytorch/fairseq/tree/master/examples/hubert",
    public_training_data="https://www.openslr.org/12",  # Link to LibriSpeech Dataset
    training_datasets={},  # "LibriSpeech": ["train"]},
    modalities=["audio"],
)
