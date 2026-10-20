from __future__ import annotations

import numpy as np

from mteb.models.modality_collators import pad_to_min_samples


def test_pad_to_min_samples_pads_only_short_clips():
    short = np.ones(368, dtype=np.float32)
    padded = pad_to_min_samples(short, 400)
    assert padded.shape == (400,)
    assert np.array_equal(padded[:368], short)
    assert not padded[368:].any()

    long = np.ones(1000, dtype=np.float32)
    assert np.array_equal(pad_to_min_samples(long, 400), long)
