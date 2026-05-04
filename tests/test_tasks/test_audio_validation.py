from __future__ import annotations

import torch

from mteb.tasks.classification._audio_validation import is_valid_audio_example
from mteb.tasks.classification.eng import vox_populi_accent_id
from mteb.tasks.classification.multilingual import vox_populi_language_id


class _FakeSamples:
    def __init__(self, data, sample_rate):
        self.data = data
        self.sample_rate = sample_rate


class _FakeAudioDecoder:
    def __init__(self, data, sample_rate=16000):
        self._data = data
        self.sample_rate = sample_rate
        self.sampling_rate = sample_rate

    def get_all_samples(self):
        return _FakeSamples(self._data, self.sample_rate)


class _FakeDataset:
    def __init__(self, rows):
        self.rows = list(rows)

    def filter(self, fn):
        return _FakeDataset([row for row in self.rows if fn(row)])

    def __len__(self):
        return len(self.rows)


def test_is_valid_audio_example_supports_audio_decoder():
    valid = {"audio": _FakeAudioDecoder(torch.ones(600))}
    too_short = {"audio": _FakeAudioDecoder(torch.ones(100))}
    non_finite = {"audio": _FakeAudioDecoder(torch.tensor([float("nan")] * 600))}

    assert is_valid_audio_example(valid)
    assert not is_valid_audio_example(too_short)
    assert not is_valid_audio_example(non_finite)


def test_vox_populi_language_id_filters_audio_decoder(monkeypatch):
    task = vox_populi_language_id.VoxPopuliLanguageID()
    task.dataset = {
        "train": _FakeDataset(
            [
                {"audio": _FakeAudioDecoder(torch.ones(600))},
                {"audio": _FakeAudioDecoder(torch.ones(100))},
            ]
        )
    }
    monkeypatch.setattr(vox_populi_language_id, "DatasetDict", lambda data: data)

    task.dataset_transform()

    assert len(task.dataset["train"]) == 1


def test_vox_populi_accent_id_filters_audio_decoder(monkeypatch):
    task = vox_populi_accent_id.VoxPopuliAccentID()
    task.dataset = {
        "test": _FakeDataset(
            [
                {"audio": _FakeAudioDecoder(torch.ones(600))},
                {"audio": _FakeAudioDecoder(torch.ones(100))},
            ]
        )
    }
    monkeypatch.setattr(vox_populi_accent_id, "DatasetDict", lambda data: data)

    task.dataset_transform()

    assert len(task.dataset["train"]) == 1
    assert len(task.dataset["test"]) == 1
