from __future__ import annotations

from collections.abc import Iterable
from functools import partial
from typing import Any

import numpy as np
import torch
import torchaudio
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, WavLMModel

from mteb.encoder_interface import AudioBatch, AudioData, PromptType
from mteb.model_meta import ModelMeta
from mteb.models.wrapper import Wrapper


class WavlmWrapper(Wrapper):
    def __init__(
        self,
        model_name: str,
        model_revision: str = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs: Any,
    ):
        self.model_name = model_name
        self.model_revision = model_revision
        self.device = device

        self.model = WavLMModel.from_pretrained(
            self.model_name, revision=self.model_revision
        ).to(self.device)

        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            self.model_name
        )
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
                    max_length=480000,
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
        raise ValueError("Wavlm models only support audio encoding.")


wavlm_base = ModelMeta(
    loader=partial(
        WavlmWrapper,
        model_name="microsoft/wavlm-base",
        model_revision="efa81aae7ff777e464159e0f877d54eac5b84f81",
    ),
    name="microsoft/wavlm-base",
    languages=["eng-Latn"],
    open_weights=True,
    revision="efa81aae7ff777e464159e0f877d54eac5b84f81",
    release_date="2022-07-19",
    max_tokens=float("inf"),
    n_parameters=94_700_000,
    memory_usage_mb=361,
    embed_dim=768,
    license="mit",
    reference="https://huggingface.co/microsoft/wavlm-base",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets={"Librispeech": ["train"]},
    modalities=["audio"],
)

wavlm_base_sd = ModelMeta(
    loader=partial(
        WavlmWrapper,
        model_name="microsoft/wavlm-base-sd",
        model_revision="fe13cca7e592cf0e11287cfede24e6999ac7dc4e",
    ),
    name="microsoft/wavlm-base-sd",
    languages=["eng-Latn"],
    open_weights=True,
    revision="fe13cca7e592cf0e11287cfede24e6999ac7dc4e",
    release_date="2022-07-19",
    max_tokens=float("inf"),
    n_parameters=94_700_000,
    memory_usage_mb=361,
    embed_dim=768,
    license="mit",
    reference="https://huggingface.co/microsoft/wavlm-base-sd",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets={"Librispeech": ["train"], "LibriMix": ["train"]},
    modalities=["audio"],
)

wavlm_base_plus = ModelMeta(
    loader=partial(
        WavlmWrapper,
        model_name="microsoft/wavlm-base-plus",
        model_revision="4c66d4806a428f2e922ccfa1a962776e232d487b",
    ),
    name="microsoft/wavlm-base-plus",
    languages=["eng-Latn"],
    open_weights=True,
    revision="4c66d4806a428f2e922ccfa1a962776e232d487b",
    release_date="2022-07-19",
    max_tokens=float("inf"),
    n_parameters=94_700_000,
    memory_usage_mb=361,
    embed_dim=768,
    license="mit",
    reference="https://huggingface.co/microsoft/wavlm-base-plus",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets={
        "Libri-Light": ["train"],
        "GigaSpeech": ["train"],
        "VoxPopuli": ["train"],
    },
    modalities=["audio"],
)

wavlm_base_plus_sv = ModelMeta(
    loader=partial(
        WavlmWrapper,
        model_name="microsoft/wavlm-base-plus-sv",
        model_revision="feb593a6c23c1cc3d9510425c29b0a14d2b07b1e",
    ),
    name="microsoft/wavlm-base-plus-sv",
    languages=["eng-Latn"],
    open_weights=True,
    revision="feb593a6c23c1cc3d9510425c29b0a14d2b07b1e",
    release_date="2022-07-19",
    max_tokens=float("inf"),
    n_parameters=94_700_000,
    memory_usage_mb=361,
    embed_dim=768,
    license="mit",
    reference="https://huggingface.co/microsoft/wavlm-base-plus-sv",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets={
        "Libri-Light": ["train"],
        "GigaSpeech": ["train"],
        "VoxPopuli": ["train"],
        "VoxCeleb1": ["train"],
    },
    modalities=["audio"],
)

wavlm_base_plus_sd = ModelMeta(
    loader=partial(
        WavlmWrapper,
        model_name="microsoft/wavlm-base-plus-sd",
        model_revision="5bd86f0662bd55704109a794c6a1b1790ea0f91a",
    ),
    name="microsoft/wavlm-base-plus-sd",
    languages=["eng-Latn"],
    open_weights=True,
    revision="5bd86f0662bd55704109a794c6a1b1790ea0f91a",
    release_date="2022-07-19",
    max_tokens=float("inf"),
    n_parameters=94_700_000,
    memory_usage_mb=361,
    embed_dim=768,
    license="mit",
    reference="https://huggingface.co/microsoft/wavlm-base-plus-sd",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets={
        "Libri-Light": ["train"],
        "GigaSpeech": ["train"],
        "VoxPopuli": ["train"],
        "LibriMix": ["train"],
    },
    modalities=["audio"],
)

wavlm_base_sv = ModelMeta(
    loader=partial(
        WavlmWrapper,
        model_name="microsoft/wavlm-base-sv",
        model_revision="0a23162ffc49adcf42bdf836a00cb2eb45af3601",
    ),
    name="microsoft/wavlm-base-sv",
    languages=["eng-Latn"],
    open_weights=True,
    revision="0a23162ffc49adcf42bdf836a00cb2eb45af3601",
    release_date="2022-07-19",
    max_tokens=float("inf"),
    n_parameters=94_700_000,
    memory_usage_mb=361,
    embed_dim=768,
    license="mit",
    reference="https://huggingface.co/microsoft/wavlm-base-sv",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets={"Librispeech": ["train"], "VoxCeleb1": ["train"]},
    modalities=["audio"],
)

wavlm_large = ModelMeta(
    loader=partial(
        WavlmWrapper,
        model_name="microsoft/wavlm-large",
        model_revision="c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c",
    ),
    name="microsoft/wavlm-large",
    languages=["eng-Latn"],
    open_weights=True,
    revision="c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c",
    release_date="2022-07-19",
    max_tokens=float("inf"),
    n_parameters=316_620_000,
    memory_usage_mb=1208,
    embed_dim=1024,
    license="mit",
    reference="https://huggingface.co/microsoft/wavlm-large",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets={
        "Libri-Light": ["train"],
        "GigaSpeech": ["train"],
        "VoxPopuli": ["train"],
    },
    modalities=["audio"],
)
