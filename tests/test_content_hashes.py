from __future__ import annotations

import numpy as np
import pytest

from mteb._content_hashes import hash_audio

_STEREO = np.random.default_rng(0).standard_normal((2, 8)).astype(np.float32)


@pytest.mark.parametrize(
    ("first", "second"),
    [
        pytest.param(
            {"array": np.zeros(16, dtype=np.float32), "sampling_rate": 16000},
            {"array": np.zeros(8, dtype=np.float64), "sampling_rate": 16000},
            id="dtype",
        ),
        pytest.param(
            {"array": _STEREO, "sampling_rate": 16000},
            {"array": _STEREO.reshape(-1), "sampling_rate": 16000},
            id="shape",
        ),
        pytest.param(
            {"array": np.full(4, ord("1"), dtype=np.uint8), "sampling_rate": 6000},
            {"array": np.full(3, ord("1"), dtype=np.uint8), "sampling_rate": 16000},
            id="sampling-rate-boundary",
        ),
    ],
)
def test_hash_audio_distinguishes_metadata(first, second) -> None:
    assert hash_audio(first) != hash_audio(second)


def test_hash_audio_is_stable_for_equal_clips() -> None:
    clip = {"array": _STEREO, "sampling_rate": 16000}
    copy = {"array": _STEREO.copy(), "sampling_rate": 16000}
    assert hash_audio(clip) == hash_audio(copy)
