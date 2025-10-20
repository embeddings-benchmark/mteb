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

from mteb.models import ModelMeta
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.models_protocols import AudioBatch
from mteb.types import Array, PromptType

COMMON_VOICE_LANGUAGES = [
    "abk-Cyrl",  # Abkhaz
    "ara-Arab",  # Arabic
    "asm-Beng",  # Assamese
    "eus-Latn",  # Basque
    "bre-Latn",  # Breton
    "cat-Latn",  # Catalan
    "zho-Hans",  # Chinese (China)
    "zho-Hant",  # Chinese (Hong Kong) + (Taiwan)
    "chv-Cyrl",  # Chuvash
    "ces-Latn",  # Czech
    "div-Thaa",  # Dhivehi
    "nld-Latn",  # Dutch
    "eng-Latn",  # English
    "epo-Latn",  # Esperanto
    "est-Latn",  # Estonian
    "fin-Latn",  # Finnish
    "fra-Latn",  # French
    "fry-Latn",  # Frisian
    "kat-Geor",  # Georgian
    "deu-Latn",  # German
    "ell-Grek",  # Greek
    "cfm-Latn",  # Hakha Chin
    "hin-Deva",  # Hindi
    "hun-Latn",  # Hungarian
    "ind-Latn",  # Indonesian
    "ina-Latn",  # Interlingua
    "gle-Latn",  # Irish
    "ita-Latn",  # Italian
    "jpn-Jpan",  # Japanese
    "kab-Latn",  # Kabyle
    "kin-Latn",  # Kinyarwanda
    "kir-Cyrl",  # Kyrgyz
    "lav-Latn",  # Latvian
    "lit-Latn",  # Lithuanian
    "lug-Latn",  # Luganda
    "mlt-Latn",  # Maltese
    "mon-Cyrl",  # Mongolian
    "ori-Orya",  # Odia
    "fas-Arab",  # Persian
    "pol-Latn",  # Polish
    "por-Latn",  # Portuguese
    "pan-Guru",  # Punjabi
    "ron-Latn",  # Romanian
    "roh-Latn",  # Romansh Sursilvan + Romansh Vallader
    "rus-Cyrl",  # Russian
    "sah-Cyrl",  # Sakha
    "slv-Latn",  # Slovenian
    "hsb-Latn",  # Upper Sorbian
    "spa-Latn",  # Spanish
    "swe-Latn",  # Swedish
    "tam-Taml",  # Tamil
    "tat-Cyrl",  # Tatar
    "tha-Thai",  # Thai
    "tur-Latn",  # Turkish
    "ukr-Cyrl",  # Ukrainian
    "vie-Latn",  # Vietnamese
    "vot-Latn",  # Votic
    "cym-Latn",  # Welsh
]


class MCTCTWrapper(AbsEncoder):
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
        self, batch: Array | Iterable[tuple[Array, str]]
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
                        if isinstance(audio, np.ndarray):
                            audio = torch.from_numpy(audio).float()
                        elif isinstance(audio, list):
                            audio = torch.tensor(audio).float()
                        else:
                            audio = audio.float()
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

    def _convert_audio(self, audio: Array) -> torch.Tensor:
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

                # Process each audio in the batch
                inputs = self.feature_extractor(
                    [audio.cpu().numpy() for audio in batch],
                    sampling_rate=self.sampling_rate,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=int(self.max_audio_length_seconds * self.sampling_rate),
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

                # Apply attention-masked pooling to exclude padding tokens
                batch_size, hidden_seq_len, hidden_size = last_hidden.shape
                device = last_hidden.device

                # Calculate proper hidden lengths based on input attention mask
                input_lengths = inputs.attention_mask.sum(dim=1)
                downsample_ratio = inputs.attention_mask.shape[1] / hidden_seq_len
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
                emb = masked_embeddings.sum(dim=1) / valid_tokens.clamp(min=1e-9)

                all_embeddings.append(emb.cpu())

        return torch.cat(all_embeddings, dim=0)

    def encode(
        self,
        inputs: AudioBatch,
        *,
        task_name: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        raise ValueError("MCTCT models only support audio encoding.")


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
