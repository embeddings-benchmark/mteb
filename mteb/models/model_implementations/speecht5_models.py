from collections.abc import Iterable
from functools import partial
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    SpeechT5ForSpeechToText,
    SpeechT5ForTextToSpeech,
    SpeechT5Processor,
)

from mteb.models import ModelMeta
from mteb.models.abs_encoder import AbsEncoder
from mteb.types import Array, PromptType


class SpeechT5Wrapper(AbsEncoder):
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

    def _process_audio(self, audio) -> list[torch.Tensor]:
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
        import torchaudio

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

    def _convert_audio_from_numpy(self, audio: Array) -> torch.Tensor:
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio)
        return audio.squeeze()

    def _load_audio_file(self, path: str) -> torch.Tensor:
        import torchaudio

        waveform, sample_rate = torchaudio.load(path)
        if sample_rate != self.sampling_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.sampling_rate)
            waveform = resampler(waveform)
        return waveform.squeeze()

    # def _pad_audio_batch(self, batch):
    #     batch = [x.reshape(-1) if x.ndim == 0 else x for x in batch]
    #     max_length = max(audio.shape[0] for audio in batch)
    #     padded_batch = [
    #         torch.nn.functional.pad(audio, (0, max_length - audio.shape[0]))
    #         for audio in batch
    #     ]
    #     return torch.stack(padded_batch)

    def get_audio_embeddings(
        self,
        audio,
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

                # batch_tensor = self._pad_audio_batch(batch) # Removed call to _pad_audio_batch
                # SpeechT5Processor expects a list of numpy arrays for audio input
                batch_arrays = [tensor.cpu().numpy() for tensor in batch]

                if batch_arrays[0].ndim == 0:
                    batch_arrays = [x.reshape(-1) for x in batch_arrays]
                elif batch_arrays[0].ndim > 1:
                    batch_arrays = [x.reshape(x.size(0), -1) for x in batch_arrays]

                inputs = self.asr_processor(
                    audio=batch_arrays,
                    sampling_rate=self.sampling_rate,
                    return_tensors="pt",
                    padding="longest",
                    truncation=True,
                    max_length=int(self.max_audio_length_s * self.sampling_rate),
                    return_attention_mask=True,
                ).to(self.device)

                outputs = self.asr_model.speecht5.encoder(
                    input_values=inputs.input_values,
                    attention_mask=inputs.attention_mask,
                )
                last_hidden = outputs.last_hidden_state

                # Apply attention-masked pooling to exclude padding tokens
                batch_size, hidden_seq_len, hidden_size = last_hidden.shape
                device = last_hidden.device

                # Calculate proper hidden lengths based on input attention mask
                input_lengths = inputs.attention_mask.sum(dim=1)
                downsample_ratio = inputs.input_values.shape[1] / hidden_seq_len
                hidden_lengths = (input_lengths.float() / downsample_ratio).long()
                hidden_lengths = torch.clamp(hidden_lengths, min=0, max=hidden_seq_len)

                # Create attention mask for hidden states
                seq_range = torch.arange(hidden_seq_len, device=device).unsqueeze(0)
                hidden_attention_mask = (seq_range < hidden_lengths.unsqueeze(1)).to(
                    last_hidden.dtype
                )

                # Apply masked mean pooling
                hidden_attention_mask = hidden_attention_mask.unsqueeze(-1)
                masked_embeddings = last_hidden * hidden_attention_mask
                valid_tokens = hidden_attention_mask.sum(dim=1)
                embeddings = masked_embeddings.sum(dim=1) / valid_tokens.clamp(min=1e-9)

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

                # Apply attention-masked pooling to exclude padding tokens
                attention_mask = (
                    inputs["attention_mask"].unsqueeze(-1).to(last_hidden.dtype)
                )
                masked_embeddings = last_hidden * attention_mask
                valid_tokens = attention_mask.sum(dim=1)
                embeddings = masked_embeddings.sum(dim=1) / valid_tokens.clamp(min=1e-9)

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
    training_datasets=set(),  # {"LibriSpeech": ["train"]},
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
    training_datasets=set(),  # {"LibriTTS": ["train"]},
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
    training_datasets=set(),
    modalities=["audio", "text"],
)
