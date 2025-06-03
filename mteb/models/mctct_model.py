from __future__ import annotations

from collections.abc import Iterable
from functools import partial
from typing import Any

import numpy as np
import torch
import torchaudio
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import MCTCTFeatureExtractor, MCTCTModel

from mteb.encoder_interface import AudioBatch, AudioData, PromptType
from mteb.model_meta import ModelMeta
from mteb.models.wrapper import Wrapper

COMMON_VOICE_LANGUAGES = [
    "ab-Cyrl",  # Abkhaz
    "ar-Arab",  # Arabic
    "as-Beng",  # Assamese
    "eu-Latn",  # Basque
    "br-Latn",  # Breton
    "ca-Latn",  # Catalan
    "zh-Hans-CN",  # Chinese (China)
    "zh-Hant-HK",  # Chinese (Hong Kong)
    "zh-Hant-TW",  # Chinese (Taiwan)
    "cv-Cyrl",  # Chuvash
    "cs-Latn",  # Czech
    "dv-Thaa",  # Dhivehi
    "nl-Latn",  # Dutch
    "en-Latn",  # English
    "eo-Latn",  # Esperanto
    "et-Latn",  # Estonian
    "fi-Latn",  # Finnish
    "fr-Latn",  # French
    "fy-Latn",  # Frisian
    "ka-Geor",  # Georgian
    "de-Latn",  # German
    "el-Grek",  # Greek
    "cfm-Latn",  # Hakha Chin
    "hi-Deva",  # Hindi
    "hu-Latn",  # Hungarian
    "id-Latn",  # Indonesian
    "ia-Latn",  # Interlingua
    "ga-Latn",  # Irish
    "it-Latn",  # Italian
    "ja-Jpan",  # Japanese
    "kab-Latn",  # Kabyle
    "rw-Latn",  # Kinyarwanda
    "ky-Cyrl",  # Kyrgyz
    "lv-Latn",  # Latvian
    "lt-Latn",  # Lithuanian
    "lg-Latn",  # Luganda
    "mt-Latn",  # Maltese
    "mn-Cyrl",  # Mongolian
    "or-Orya",  # Odia
    "fa-Arab",  # Persian
    "pl-Latn",  # Polish
    "pt-Latn",  # Portuguese
    "pa-Guru",  # Punjabi
    "ro-Latn",  # Romanian
    "rm-Surs",  # Romansh Sursilvan
    "rm-Vall",  # Romansh Vallader
    "ru-Cyrl",  # Russian
    "sah-Cyrl",  # Sakha
    "sl-Latn",  # Slovenian
    "hsb-Latn",  # Upper Sorbian
    "es-Latn",  # Spanish
    "sv-Latn",  # Swedish
    "ta-Taml",  # Tamil
    "tt-Cyrl",  # Tatar
    "th-Thai",  # Thai
    "tr-Latn",  # Turkish
    "uk-Cyrl",  # Ukrainian
    "vi-Latn",  # Vietnamese
    "vot-Latn",  # Votic
    "cy-Latn",  # Welsh
]


class MCTCTWrapper(Wrapper):
    def __init__(
        self,
        model_name: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs: Any,
    ):
        self.model_name = model_name
        self.device = device

        self.model = MCTCTModel.from_pretrained(model_name).to(device)
        self.feature_extractor = MCTCTFeatureExtractor.from_pretrained(model_name)
        self.sampling_rate = self.feature_extractor.sampling_rate  # 16000 Hz

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

        # Ensure float type
        audio = audio.float()

        # Convert to mono if needed (MCTCT expects mono audio)
        if audio.dim() > 1 and audio.shape[0] > 1:  # If multi-channel
            audio = torch.mean(audio, dim=0, keepdim=True)  # Convert to mono

        return audio.squeeze()

    def _load_audio_file(self, path: str) -> torch.Tensor:
        waveform, sample_rate = torchaudio.load(path)

        # Convert to mono if needed
        if waveform.shape[0] > 1:  # If multi-channel
            waveform = torch.mean(waveform, dim=0, keepdim=True)  # Convert to mono

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

                # Process each audio in the batch
                inputs = self.feature_extractor(
                    [audio.cpu().numpy() for audio in batch],
                    sampling_rate=self.sampling_rate,
                    return_tensors="pt",
                    padding=True,
                ).to(self.device)

                # Get embeddings from the model
                outputs = self.model(
                    input_features=inputs.input_features,
                    attention_mask=inputs.attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )

                # Get embeddings from the final layer hidden states
                last_hidden = outputs.hidden_states[-1]
                emb = last_hidden.mean(dim=1).cpu()
                all_embeddings.append(emb)

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


mctct_large = ModelMeta(
    loader=partial(
        MCTCTWrapper,
        model_name="speechbrain/m-ctc-t-large",
    ),
    name="speechbrain/m-ctc-t-large",
    languages=COMMON_VOICE_LANGUAGES,  # Supports 60 languages
    open_weights=True,
    revision="ed014c8255cea2c36f87a71cf2533b665ba00863",
    release_date="2022-01-10",
    max_tokens=None,
    n_parameters=1_058_978_691,
    memory_usage_mb=4039,
    embed_dim=1536,
    license="apache-2.0",
    reference="https://huggingface.co/speechbrain/m-ctc-t-large",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=False,
    public_training_code="https://github.com/speechbrain/speechbrain",
    public_training_data="https://github.com/speechbrain/speechbrain",
    training_datasets={"Common Voice": ["train"], "VoxPopuli": ["train"]},
    modalities=["audio"],
)
