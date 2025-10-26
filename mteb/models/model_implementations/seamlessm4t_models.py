from collections.abc import Iterable
from functools import partial
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, SeamlessM4Tv2Model

from mteb.models import ModelMeta
from mteb.models.abs_encoder import AbsEncoder
from mteb.types import Array, PromptType


class SeamlessM4TWrapper(AbsEncoder):
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

        self.model = SeamlessM4Tv2Model.from_pretrained(model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.sampling_rate = self.processor.feature_extractor.sampling_rate

        self.speech_encoder = self.model.speech_encoder

        self.model = self.model.to(device)
        self.speech_encoder = self.speech_encoder.to(device)

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
        return audio.squeeze()

    def _load_audio_file(self, path: str) -> torch.Tensor:
        import torchaudio

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
        audio,
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
                # batch_tensor = self._pad_audio_batch(batch)

                max_samples = int(self.max_audio_length_seconds * self.sampling_rate)
                truncated_audio = []
                for w in batch:
                    audio_np = w.cpu().numpy()
                    if len(audio_np) > max_samples:
                        audio_np = audio_np[:max_samples]
                    truncated_audio.append(audio_np)

                inputs = self.processor(
                    audios=truncated_audio,
                    sampling_rate=self.sampling_rate,
                    return_tensors="pt",
                    padding=True,
                )

                input_features = inputs.input_features.to(self.device)
                attention_mask = (
                    inputs.attention_mask.to(self.device)
                    if hasattr(inputs, "attention_mask")
                    else None
                )

                outputs = self.speech_encoder(
                    input_features,
                    attention_mask=attention_mask,
                    output_hidden_states=False,
                )

                last_hidden_state = outputs.last_hidden_state

                # Apply attention-masked pooling to exclude padding tokens
                if attention_mask is not None:
                    batch_size, hidden_seq_len, hidden_size = last_hidden_state.shape
                    device = last_hidden_state.device

                    # For SeamlessM4T, check if attention mask matches hidden state length
                    if attention_mask.shape[1] != hidden_seq_len:
                        # Calculate downsample ratio and proper hidden lengths
                        input_lengths = attention_mask.sum(dim=1)
                        downsample_ratio = attention_mask.shape[1] / hidden_seq_len
                        hidden_lengths = (
                            input_lengths.float() / downsample_ratio
                        ).long()
                        hidden_lengths = torch.clamp(
                            hidden_lengths, min=0, max=hidden_seq_len
                        )

                        # Create attention mask for hidden states
                        seq_range = torch.arange(
                            hidden_seq_len, device=device
                        ).unsqueeze(0)
                        hidden_attention_mask = (
                            seq_range < hidden_lengths.unsqueeze(1)
                        ).to(last_hidden_state.dtype)
                    else:
                        # Use the attention mask directly if dimensions match
                        hidden_attention_mask = attention_mask.to(
                            last_hidden_state.dtype
                        )

                    # Apply masked mean pooling
                    hidden_attention_mask = hidden_attention_mask.unsqueeze(-1)
                    masked_embeddings = last_hidden_state * hidden_attention_mask
                    valid_tokens = hidden_attention_mask.sum(dim=1)
                    embeddings = masked_embeddings.sum(dim=1) / valid_tokens.clamp(
                        min=1e-9
                    )
                else:
                    # Fallback to simple mean pooling if no attention mask
                    embeddings = last_hidden_state.mean(dim=1)

                all_embeddings.append(embeddings.cpu())

        if all_embeddings:
            return torch.cat(all_embeddings, dim=0)
        else:
            return torch.zeros((0, self.model.config.hidden_size))

    def encode(
        self,
        inputs,
        *,
        task_name: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        raise ValueError("SeamlessM4T models only support audio encoding.")


seamless_m4t_v2_large = ModelMeta(
    loader=partial(
        SeamlessM4TWrapper,
        model_name="facebook/seamless-m4t-v2-large",
        max_audio_length_seconds=5.0,  # Conservative default for memory efficiency
    ),
    name="facebook/seamless-m4t-v2-large",
    languages=[
        "eng-Latn"
    ],  # multilingual: supported languages can be found in the reference
    open_weights=True,
    revision="5f8cc790b19fc3f67a61c105133b20b34e3dcb76",
    release_date="2023-11-06",
    max_tokens=None,
    n_parameters=2_300_000_000,
    memory_usage_mb=8809,
    embed_dim=1024,
    license="mit",
    reference="https://huggingface.co/facebook/seamless-m4t-v2-large",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=False,
    public_training_code="https://github.com/facebookresearch/seamless_communication",
    public_training_data=None,
    training_datasets=None,
    modalities=["audio"],
)
