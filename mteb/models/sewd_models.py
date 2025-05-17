from __future__ import annotations

from collections.abc import Iterable
from functools import partial
from typing import Any

import numpy as np
import torch
import torchaudio
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import SEWDForCTC, Wav2Vec2FeatureExtractor

from mteb.encoder_interface import AudioBatch, AudioData, PromptType
from mteb.model_meta import ModelMeta
from mteb.models.wrapper import Wrapper


class SewDWrapper(Wrapper):
    def __init__(
        self,
        model_name: str = "asapp/sew-d-base-plus-400k-ft-ls100h",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs: Any,
    ):
        self.model_name = model_name
        self.device = device
        # SewD uses the same feature extractor as Wav2Vec2
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = SEWDForCTC.from_pretrained(model_name).to(self.device)
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

        if isinstance(batch, tuple):
            for audio, _ in batch:
                waveforms.append(self._convert_audio(audio))
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
                        waveforms.append(self._convert_audio(audio))
                    elif "path" in item:
                        waveforms.append(self._load_audio_file(item["path"]))
                elif isinstance(item, (np.ndarray, torch.Tensor)):
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

    def _pad_audio_batch(self, batch: list[torch.Tensor]) -> torch.Tensor:
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
        hidden_layer: float = 1.0,
        **kwargs: Any,
    ) -> torch.Tensor:
        processed_audio = self._process_audio(audio)
        all_embeddings = []

        with torch.no_grad():
            for i in tqdm(range(0, len(processed_audio), batch_size)):
                batch = processed_audio[i : i + batch_size]
                batch = self._pad_audio_batch(batch)

                inputs = self.feature_extractor(
                    batch,
                    sampling_rate=self.sampling_rate,
                    return_tensors="pt",
                    padding="longest",
                    return_attention_mask=True,
                ).to(self.device)

                outputs = self.model(
                    inputs.input_features,
                    output_hidden_states=True,
                )

                hidden_states = outputs.hidden_states
                no_hidden_states = len(hidden_states)

                layer_idx = int(hidden_layer * no_hidden_states)
                layer_idx = min(max(layer_idx, 1), no_hidden_states) - 1

                selected_hidden = hidden_states[layer_idx]
                embeddings = torch.mean(selected_hidden, dim=1)

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


# Model Metas for Different Whisper Models

sewd_base = ModelMeta(
    loader=partial(SewDWrapper, model_name="openai/whisper-tiny"),
    name="asapp/sew-d-base-plus-400k-ft-ls100h",
    languages=["eng"],
    open_weights=True,
    revision="d78e7a1b50e9f1ce21887ca069330efdd5ccd4ca",
    release_date="2021-09-14",
    max_tokens=float("inf"),
    n_parameters=95_000_000,
    memory_usage_mb=675,
    embed_dim=768,
    license="apache-2.0",
    reference="https://huggingface.co/asapp/sew-d-base-plus-400k-ft-ls100h",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets={"LibriSpeech": ["train"]},
    modalities=["audio"],
)

sewd_tiny = ModelMeta(
    loader=partial(
        SewDWrapper,
        model_name="asapp/sew-d-tiny-100k-ft-ls100h",
    ),
    name="asapp/sew-d-tiny-100k-ft-ls100h",
    languages=["eng"],
    open_weights=True,
    revision="1966cdcfbd2123ee90b003c2aa6ec6fe204cc4d8",
    release_date="2021-09-14",
    max_tokens=float("inf"),
    n_parameters=19_700_000,
    memory_usage_mb=92,
    embed_dim=256,
    license="apache-2.0",
    reference="https://huggingface.co/asapp/sew-d-tiny-100k-ft-ls100h",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets={"LibriSpeech": ["train"]},
    modalities=["audio"],
)

sewd_mid = ModelMeta(
    loader=partial(
        SewDWrapper,
        model_name="asapp/sew-d-mid-400k-ft-ls100h",
    ),
    name="asapp/sew-d-mid-400k-ft-ls100h",
    languages=["eng"],
    open_weights=True,
    revision="b2ff9fdb3bddc81657cf5f16bc0c510be0a39b3e",
    release_date="2021-09-14",
    max_tokens=float("inf"),
    n_parameters=139_000_000,
    memory_usage_mb=530,
    embed_dim=768,
    license="apache-2.0",
    reference="https://huggingface.co/asapp/sew-d-mid-400k-ft-ls100h",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets={"LibriSpeech": ["train"]},
    modalities=["audio"],
)
