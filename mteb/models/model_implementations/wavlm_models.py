from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, WavLMModel

from mteb.models import ModelMeta
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.models_protocols import AudioBatch
from mteb.types import Array, PromptType


class WavlmWrapper(AbsEncoder):
    def __init__(
        self,
        model_name: str,
        model_revision: str | None = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_audio_length_seconds: float = 30.0,
        **kwargs: Any,
    ):
        self.model_name = model_name
        self.model_revision = model_revision
        self.device = device
        self.max_audio_length_seconds = max_audio_length_seconds

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
                        elif isinstance(audio, list):
                            audio = torch.tensor(audio).float()
                        else:
                            audio = audio.float()
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

                # Let feature extractor handle all padding
                batch_numpy = [
                    b.cpu().numpy() if isinstance(b, torch.Tensor) else b for b in batch
                ]

                inputs = self.feature_extractor(
                    batch_numpy,
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

                last_hidden_state = outputs.hidden_states[-1]

                # Apply attention-masked pooling to exclude padding tokens
                batch_size, hidden_seq_len, hidden_size = last_hidden_state.shape
                device = last_hidden_state.device

                # Calculate proper hidden lengths based on input attention mask
                input_lengths = inputs.attention_mask.sum(dim=1)
                downsample_ratio = inputs.input_values.shape[1] / hidden_seq_len
                hidden_lengths = (input_lengths.float() / downsample_ratio).long()
                hidden_lengths = torch.clamp(hidden_lengths, min=0, max=hidden_seq_len)

                # Create attention mask for hidden states
                seq_range = torch.arange(hidden_seq_len, device=device).unsqueeze(0)
                hidden_attention_mask = (seq_range < hidden_lengths.unsqueeze(1)).to(
                    last_hidden_state.dtype
                )

                # Apply masked mean pooling
                hidden_attention_mask = hidden_attention_mask.unsqueeze(-1)
                masked_embeddings = last_hidden_state * hidden_attention_mask
                valid_tokens = hidden_attention_mask.sum(dim=1)
                embeddings = masked_embeddings.sum(dim=1) / valid_tokens.clamp(min=1e-9)

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
    loader=WavlmWrapper,
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
    training_datasets={"Librispeech"},
    modalities=["audio"],
)

wavlm_base_sd = ModelMeta(
    loader=WavlmWrapper,
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
    training_datasets={"Librispeech", "LibriMix"},
    modalities=["audio"],
)

wavlm_base_plus = ModelMeta(
    loader=WavlmWrapper,
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
        "Libri-Light",
        "GigaSpeech",
        "VoxPopuli",
    },
    modalities=["audio"],
)

wavlm_base_plus_sv = ModelMeta(
    loader=WavlmWrapper,
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
        "Libri-Light",
        "GigaSpeech",
        "VoxPopuli",
        "VoxCeleb1",
    },
    modalities=["audio"],
)

wavlm_base_plus_sd = ModelMeta(
    loader=WavlmWrapper,
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
        "Libri-Light",
        "GigaSpeech",
        "VoxPopuli",
        "LibriMix",
    },
    modalities=["audio"],
)

wavlm_base_sv = ModelMeta(
    loader=WavlmWrapper,
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
    training_datasets={"Librispeech", "VoxCeleb1"},
    modalities=["audio"],
)

wavlm_large = ModelMeta(
    loader=WavlmWrapper,
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
        "Libri-Light",
        "GigaSpeech",
        "VoxPopuli",
    },
    modalities=["audio"],
)
