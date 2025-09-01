from __future__ import annotations

from collections.abc import Iterable
from functools import partial
from typing import Any

import numpy as np
import torch
import torchaudio
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import WhisperModel, WhisperProcessor

from mteb.encoder_interface import AudioBatch, AudioData, PromptType
from mteb.model_meta import ModelMeta
from mteb.models.wrapper import Wrapper


class WhisperAudioWrapper(Wrapper):
    def __init__(
        self,
        model_name: str = "openai/whisper-tiny",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        revision: str = "main",
        **kwargs: Any,
    ):
        self.model_name = model_name
        self.device = device
        self.revision = revision

        self.model = WhisperModel.from_pretrained(model_name, revision=revision).to(
            device
        )
        self.processor = WhisperProcessor.from_pretrained(model_name, revision=revision)
        self.sampling_rate = self.processor.feature_extractor.sampling_rate

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
                batch_arrays = [tensor.numpy() for tensor in batch]

                inputs = self.processor(
                    batch_arrays,
                    sampling_rate=self.sampling_rate,
                    return_tensors="pt",
                    padding="max_length",
                    max_length=None,
                    return_attention_mask=True,
                ).to(self.device)

                outputs = self.model.encoder(
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
        raise ValueError("Whisper models only support audio encoding.")


# Model Metas for Different Whisper Models
whisper_langs = [
    "eng-Latn",
    "zho-Hans",
    "deu-Latn",
    "spa-Latn",
    "rus-Cyrl",
    "kor-Hang",
    "fra-Latn",
    "jpn-Jpan",
    "por-Latn",
    "tur-Latn",
    "pol-Latn",
    "cat-Latn",
    "nld-Latn",
    "ara-Arab",
    "swe-Latn",
    "ita-Latn",
    "ind-Latn",
    "hin-Deva",
    "fin-Latn",
    "vie-Latn",
    "heb-Hebr",
    "ukr-Cyrl",
    "ell-Grek",
    "msa-Latn",
    "ces-Latn",
    "ron-Latn",
    "dan-Latn",
    "hun-Latn",
    "tam-Taml",
    "nob-Latn",
    "tha-Thai",
    "urd-Arab",
    "hrv-Latn",
    "bul-Cyrl",
    "lit-Latn",
    "lat-Latn",
    "mri-Latn",
    "mal-Mlym",
    "cym-Latn",
    "slk-Latn",
    "tel-Telu",
    "fas-Arab",
    "lav-Latn",
    "ben-Beng",
    "srp-Cyrl",
    "aze-Latn",
    "slv-Latn",
    "kan-Knda",
    "est-Latn",
    "mkd-Cyrl",
    "bre-Latn",
    "eus-Latn",
    "isl-Latn",
    "hye-Armn",
    "nep-Deva",
    "mon-Cyrl",
    "bos-Latn",
    "kaz-Cyrl",
    "sqi-Latn",
    "swa-Latn",
    "glg-Latn",
    "mar-Deva",
    "pan-Guru",
    "sin-Sinh",
    "khm-Khmr",
    "sna-Latn",
    "yor-Latn",
    "som-Latn",
    "afr-Latn",
    "oci-Latn",
    "kat-Geor",
    "bel-Cyrl",
    "tgk-Cyrl",
    "snd-Arab",
    "guj-Gujr",
    "amh-Ethi",
    "yid-Hebr",
    "lao-Laoo",
    "uzb-Latn",
    "fao-Latn",
    "hat-Latn",
    "pus-Arab",
    "tuk-Latn",
    "nno-Latn",
    "mlt-Latn",
    "san-Deva",
    "ltz-Latn",
    "mya-Mymr",
    "bod-Tibt",
    "tgl-Latn",
    "mlg-Latn",
    "asm-Beng",
    "tat-Cyrl",
    "haw-Latn",
    "lin-Latn",
    "hau-Latn",
    "bak-Cyrl",
    "jav-Latn",
    "sun-Latn",
]


whisper_tiny = ModelMeta(
    loader=partial(WhisperAudioWrapper, model_name="openai/whisper-tiny"),
    name="openai/whisper-tiny",
    languages=whisper_langs,
    open_weights=True,
    revision="main",
    release_date="2022-09-27",
    max_tokens=float("inf"),
    n_parameters=39_000_000,
    memory_usage_mb=144,
    embed_dim=512,
    license="mit",
    reference="https://huggingface.co/openai/whisper-tiny",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
    modalities=["audio"],
)

whisper_base = ModelMeta(
    loader=partial(WhisperAudioWrapper, model_name="openai/whisper-base"),
    name="openai/whisper-base",
    languages=whisper_langs,
    open_weights=True,
    revision="main",
    release_date="2022-09-27",
    max_tokens=float("inf"),
    n_parameters=74_000_000,
    memory_usage_mb=277,
    embed_dim=512,
    license="mit",
    reference="https://huggingface.co/openai/whisper-base",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
    modalities=["audio"],
)

whisper_small = ModelMeta(
    loader=partial(WhisperAudioWrapper, model_name="openai/whisper-small"),
    name="openai/whisper-small",
    languages=whisper_langs,
    open_weights=True,
    revision="main",
    release_date="2022-09-27",
    max_tokens=float("inf"),
    n_parameters=244_000_000,
    memory_usage_mb=922,
    embed_dim=768,
    license="mit",
    reference="https://huggingface.co/openai/whisper-small",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
    modalities=["audio"],
)

whisper_medium = ModelMeta(
    loader=partial(WhisperAudioWrapper, model_name="openai/whisper-medium"),
    name="openai/whisper-medium",
    languages=whisper_langs,
    open_weights=True,
    revision="main",
    release_date="2022-09-27",
    max_tokens=float("inf"),
    n_parameters=769_000_000,
    memory_usage_mb=2914,
    embed_dim=1024,
    license="mit",
    reference="https://huggingface.co/openai/whisper-medium",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
    modalities=["audio"],
)

whisper_large_v3 = ModelMeta(
    loader=partial(WhisperAudioWrapper, model_name="openai/whisper-large-v3"),
    name="openai/whisper-large-v3",
    languages=whisper_langs,
    open_weights=True,
    revision="main",
    release_date="2022-09-27",
    max_tokens=float("inf"),
    n_parameters=1_550_000_000,
    memory_usage_mb=5887,
    embed_dim=1280,
    license="mit",
    reference="https://huggingface.co/openai/whisper-large-v3",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
    modalities=["audio"],
)
