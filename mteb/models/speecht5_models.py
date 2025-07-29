from __future__ import annotations

from collections.abc import Iterable
from functools import partial
from typing import Any

import numpy as np
import torch
import torchaudio
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    SpeechT5ForSpeechToText,
    SpeechT5ForTextToSpeech,
    SpeechT5Processor,
)

from mteb.encoder_interface import AudioBatch, AudioData, PromptType
from mteb.model_meta import ModelMeta
from mteb.models.wrapper import Wrapper


class SpeechT5Wrapper(Wrapper):
    def __init__(
        self,
        model_name: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs: Any,
    ):
        self.model_name = model_name
        self.device = device

        self.asr_processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_asr")
        self.tts_processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")

        # Initialize models for both audio and text encoding
        self.asr_model = SpeechT5ForSpeechToText.from_pretrained(
            "microsoft/speecht5_asr"
        ).to(self.device)
        self.tts_model = SpeechT5ForTextToSpeech.from_pretrained(
            "microsoft/speecht5_tts"
        ).to(self.device)

        self.sampling_rate = self.asr_processor.feature_extractor.sampling_rate

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
                        if isinstance(audio, np.ndarray):
                            audio = torch.from_numpy(audio).float()
                        elif isinstance(audio, torch.Tensor):
                            audio = audio.float()
                        elif isinstance(audio, list):
                            audio = torch.tensor(audio, dtype=torch.float32)
                        else:
                            raise TypeError(f"Unsupported audio type: {type(audio)}")
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
        hidden_layer: float = 1.0,
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

                batch_tensor = self._pad_audio_batch(batch)

                if batch_tensor.ndim == 1:
                    batch_tensor = batch_tensor.unsqueeze(0)
                elif batch_tensor.ndim > 2:
                    batch_tensor = batch_tensor.view(batch_tensor.size(0), -1)

                inputs = self.asr_processor(
                    audio=batch_tensor.cpu().numpy(),
                    sampling_rate=self.sampling_rate,
                    return_tensors="pt",
                    padding="longest",
                    truncation=True,
                    max_length=30 * self.sampling_rate,  # 30 seconds max
                    return_attention_mask=True,
                ).to(self.device)

                outputs = self.asr_model.speecht5.encoder(
                    input_values=inputs.input_values,
                    attention_mask=inputs.attention_mask,
                )
                last_hidden = outputs.last_hidden_state
                embeddings = torch.mean(last_hidden, dim=1)

                all_embeddings.append(embeddings.cpu())

        if all_embeddings:
            return torch.cat(all_embeddings, dim=0)
        else:
            return torch.zeros((0, self.asr_model.config.hidden_size))

    def get_text_embeddings(
        self,
        texts: list[str],
        *,
        task_name: str | None = None,
        prompt_type: PromptType | None = None,
        batch_size: int = 4,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Get text embeddings using the text encoder."""
        all_embeddings = []

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]

                # Process text through tokenizer
                inputs = self.tts_processor(
                    text=batch_texts,
                    return_tensors="pt",
                    padding="longest",
                    truncation=True,
                ).to(self.device)

                outputs = self.tts_model.speecht5.encoder(
                    input_values=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                )

                last_hidden = outputs.last_hidden_state
                embeddings = torch.mean(last_hidden, dim=1)
                all_embeddings.append(embeddings.cpu())

        if all_embeddings:
            return torch.cat(all_embeddings, dim=0)
        else:
            return torch.zeros((0, self.tts_model.config.hidden_size))

    def encode(
        self,
        inputs: list[str],
        *,
        task_name: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        return self.get_text_embeddings(inputs, **kwargs).numpy()


# ASR model - Optimized for Speech Recognition tasks
speecht5_asr = ModelMeta(
    loader=partial(
        SpeechT5Wrapper,
        model_name="microsoft/speecht5_asr",
    ),
    name="microsoft/speecht5_asr",
    languages=["eng-Latn"],
    open_weights=True,
    revision="53615c10408485422e09a12cda191a747f4bbe34",
    release_date="2022-05-16",
    max_tokens=None,
    n_parameters=151_575_936,
    memory_usage_mb=578,
    embed_dim=768,
    license="mit",
    reference="https://huggingface.co/microsoft/speecht5_asr",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=False,
    public_training_code="https://github.com/microsoft/SpeechT5",
    public_training_data="https://www.openslr.org/12",
    training_datasets={},  # {"LibriSpeech": ["train"]},
    modalities=["audio"],
)

# TTS model - Optimized for Text-to-Speech tasks
speecht5_tts = ModelMeta(
    loader=partial(
        SpeechT5Wrapper,
        model_name="microsoft/speecht5_tts",
    ),
    name="microsoft/speecht5_tts",
    languages=["eng-Latn"],
    open_weights=True,
    revision="30fcde30f19b87502b8435427b5f5068e401d5f6",
    release_date="2022-05-16",
    max_tokens=None,
    n_parameters=146_335_465,
    memory_usage_mb=558,
    embed_dim=768,
    license="mit",
    reference="https://huggingface.co/microsoft/speecht5_tts",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=False,
    public_training_code="https://github.com/microsoft/SpeechT5",
    public_training_data="https://www.openslr.org/12",
    training_datasets={},  # {"LibriTTS": ["train"]},
    modalities=["text"],
)

# Voice Conversion model - Optimized for Speech-to-Speech conversion tasks
speecht5_multimodal = ModelMeta(
    loader=partial(
        SpeechT5Wrapper,
        model_name="microsoft/speecht5_multimodal",
    ),
    name="microsoft/speecht5_multimodal",
    languages=["eng-Latn"],
    open_weights=True,
    revision="N/A",
    release_date="2022-05-16",
    max_tokens=None,
    n_parameters=297_911_401,  # Combined ASR + TTS parameters
    memory_usage_mb=1136,  # Combined memory usage
    embed_dim=768,
    license="mit",
    reference="https://huggingface.co/microsoft/speecht5_asr",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=False,
    public_training_code="https://github.com/microsoft/SpeechT5",
    public_training_data="http://www.festvox.org/cmu_arctic/",
    training_datasets={},
    modalities=["audio", "text"],
)
