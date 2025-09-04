from __future__ import annotations

import io
from typing import List, Optional

import numpy as np
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence


def get_audio_transform(sample_rate: int, target_sampling_rate: int):
    if sample_rate == target_sampling_rate:
        return None
    return torchaudio.transforms.Resample(
        orig_freq=sample_rate, new_freq=target_sampling_rate
    )


class AudioDataset(torch.utils.data.Dataset):
    """HF dataset -> waveform tensor. Handles resampling and mono conversion."""

    def __init__(
        self,
        hf_dataset,
        audio_column_name: str = "audio",
        target_sampling_rate: int | None = None,
        mono: bool = True,
        transform=None,
    ):
        self.dataset = hf_dataset
        self.transform = transform
        self.audio_column_name = audio_column_name
        self.target_sampling_rate = target_sampling_rate
        self.mono = mono

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        audio = self.dataset[idx][self.audio_column_name]
        waveform = None
        sample_rate = None

        # Support bytes, file path, or already tensor-like
        if isinstance(audio, (bytes, bytearray)):
            waveform, sample_rate = torchaudio.load(io.BytesIO(audio))
        elif isinstance(audio, str):
            waveform, sample_rate = torchaudio.load(audio)
        else:
            # If HF 'audio' object, it might be a dict with 'array' and 'sampling_rate'
            if isinstance(audio, dict) and "array" in audio:
                waveform = torch.tensor(audio["array"])
                sample_rate = audio.get("sampling_rate", None)
                if waveform.dim() == 1:
                    waveform = waveform.unsqueeze(0)  # (1, T)
            elif isinstance(audio, torch.Tensor):
                waveform = audio
                sample_rate = (
                    self.target_sampling_rate
                )  # Assume if already tensor, it's at target rate
            elif isinstance(audio, np.ndarray):
                waveform = torch.from_numpy(audio)
                sample_rate = (
                    self.target_sampling_rate
                )  # Assume if already numpy, it's at target rate

        if waveform is None:
            raise ValueError(f"Unsupported audio format: {type(audio)}")

        # Ensure waveform is float32 tensor shaped (channels, time)
        if waveform.dtype != torch.float32:
            waveform = waveform.to(torch.float32)

        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # Ensure (C, T)

        # Handle resampling if target_sampling_rate is provided and different
        if (
            self.target_sampling_rate is not None
            and sample_rate is not None
            and sample_rate != self.target_sampling_rate
        ):
            resampler = get_audio_transform(sample_rate, self.target_sampling_rate)
            if resampler:  # resampler can be None if rates are same
                waveform = resampler(waveform)
                sample_rate = self.target_sampling_rate

        # Convert to mono if required and multi-channel
        if self.mono and waveform.size(0) > 1:
            waveform = waveform.mean(
                dim=0, keepdim=True
            )  # Averages channels, results in (1, T)

        # Apply additional transform
        if self.transform is not None:
            waveform = self.transform(waveform)

        return waveform  # shape: (channels, time) - which should now be (1, T) if mono=True


class CustomAudioCollate:
    """Pad (and optionally truncate) a batch of waveforms to a consistent length.

    Args:
        max_length_samples: if provided, pad/truncate to this length.
            If None -> pad to longest sample in the batch.
        mono: if True, collapses channels to mono by averaging.
        pad_value: value to use for padding (0.0 recommended).
    """

    def __init__(
        self,
        max_length_samples: Optional[int] = None,
        mono: bool = True,
        pad_value: float = 0.0,
    ):
        self.max_length_samples = max_length_samples
        self.mono = mono
        self.pad_value = pad_value

    def __call__(self, batch: List[torch.Tensor]):
        # batch: list of tensors (C, T), where C should now be 1 due to AudioDataset handling mono conversion
        waveforms_processed_for_pad_sequence = []
        lengths = []
        for w in batch:
            # Ensure (C, T) and then make (T,) for pad_sequence
            if w.dim() == 1:
                w = w.unsqueeze(0)  # ensure (C, T) if it somehow became (T,)
            w = w.squeeze(0)  # Now it's (T,), ready for pad_sequence

            waveforms_processed_for_pad_sequence.append(
                w.float()
            )  # Ensure float and it's (T,)
            lengths.append(waveforms_processed_for_pad_sequence[-1].shape[-1])

        # pad to longest in batch (or to global max_length_samples if smaller)
        max_in_batch = max(lengths)
        target_len = (
            max_in_batch
            if self.max_length_samples is None
            else min(max_in_batch, self.max_length_samples)
        )

        # Pad or truncate within pad_sequence to target_len
        padded = pad_sequence(
            [w[:target_len] for w in waveforms_processed_for_pad_sequence],
            batch_first=True,
            padding_value=self.pad_value,
        )

        # If global max_length_samples is set and current padded length is shorter, pad more:
        if (
            self.max_length_samples is not None
            and padded.shape[1] < self.max_length_samples
        ):
            pad_amt = self.max_length_samples - padded.shape[1]
            padded = torch.nn.functional.pad(padded, (0, pad_amt), value=self.pad_value)
        # If padded length is greater than max_length_samples, truncate
        elif (
            self.max_length_samples is not None
            and padded.shape[1] > self.max_length_samples
        ):
            padded = padded[:, : self.max_length_samples]

        # output shape: (batch_size, T)
        # return waveforms as (batch_size, 1, T) to be consistent with audio models expecting channel dim
        padded = padded.unsqueeze(1)  # (B, 1, T)
        lengths = torch.tensor(lengths, dtype=torch.long)
        return {"waveforms": padded, "lengths": lengths}
