from __future__ import annotations

from collections.abc import Iterable
from functools import partial
from typing import Any

import numpy as np
import torch
import torchaudio
from torch.utils.data import DataLoader
from tqdm import tqdm

from mteb.encoder_interface import AudioBatch, AudioData, PromptType
from mteb.model_meta import ModelMeta
from mteb.models.wrapper import Wrapper
from mteb.requires_package import requires_package


class CNN14Wrapper(Wrapper):
    def __init__(
        self,
        model_name: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_audio_length_s: float = 30.0,
        **kwargs: Any,
    ):
        self.model_name = model_name
        self.device = device
        self.max_audio_length_s = max_audio_length_s

        requires_package(
            self,
            "speechbrain",
            "speechbrain/cnn14-esc50",
            "pip install 'mteb[speechbrain]'",
        )

        from speechbrain.inference.classifiers import AudioClassifier

        # Load the SpeechBrain model
        self.model = AudioClassifier.from_hparams(
            source=model_name,
            savedir="pretrained_models/cnn14-esc50",
            run_opts={"device": device},
        )

        # SpeechBrain uses a 16kHz sampling rate for audio
        self.sampling_rate = 16000

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

    def _convert_audio(self, audio: AudioData) -> torch.Tensor:
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio)
        audio = audio.squeeze()

        # Apply audio truncation (configurable limit)
        max_length = int(self.max_audio_length_s * self.sampling_rate)
        if audio.shape[-1] > max_length:
            audio = audio[..., :max_length]

        return audio

    def _load_audio_file(self, path: str) -> torch.Tensor:
        waveform, sample_rate = torchaudio.load(path)
        if sample_rate != self.sampling_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.sampling_rate)
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

                # Convert batch to tensors and move to device
                batch_tensor = self._pad_audio_batch(batch).to(self.device)

                feats = self.model.mods.compute_features(batch_tensor)
                B, F, T = feats.shape
                if F < 64 or T < 80:
                    # zero‐pad in the frequency or time dimension until it's at least [64, 80]:
                    pad_freq = max(0, 64 - F)
                    pad_time = max(0, 80 - T)
                    feats = torch.nn.functional.pad(feats, (0, pad_time, 0, pad_freq))
                embeddings = self.model.mods.embedding_model(feats)
                # Apply mean pooling over time dimension if needed
                if embeddings.dim() > 2:
                    embeddings = torch.mean(embeddings, dim=1)

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
        raise ValueError("CNN14 models only support audio encoding.")


cnn14_esc50 = ModelMeta(
    loader=partial(
        CNN14Wrapper,
        model_name="speechbrain/cnn14-esc50",
    ),
    name="speechbrain/cnn14-esc50",
    languages=["eng-Latn"],
    open_weights=True,
    revision="422a112e9a22a5fac0d37571aacaee5caf154395",
    release_date="2022-11-26",
    max_tokens=None,
    n_parameters=80_753_615,
    memory_usage_mb=308,
    embed_dim=2048,
    license="apache-2.0",
    reference="https://huggingface.co/speechbrain/cnn14-esc50",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=False,
    public_training_code="https://github.com/speechbrain/speechbrain",
    public_training_data=None,
    training_datasets=None,  # ["ESC-50", "VGGSound"],
    modalities=["audio"],
)
