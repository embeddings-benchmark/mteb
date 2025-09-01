from __future__ import annotations

import io

import torch
import torchaudio


class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, audio_column_name: str = "audio", transform=None):
        self.dataset = hf_dataset
        self.transform = transform
        self.audio_column_name = audio_column_name

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        audio = self.dataset[idx][self.audio_column_name]
        if isinstance(audio, bytes):
            waveform, sample_rate = torchaudio.load(io.BytesIO(audio))
        elif isinstance(audio, str):
            # Assuming audio is a file path
            waveform, sample_rate = torchaudio.load(audio)
        else:
            # Assume audio is already a tensor or in a usable format
            waveform = audio
        if self.transform is not None:
            waveform = self.transform(waveform)
        return waveform


def custom_collate_fn(batch):
    return batch
