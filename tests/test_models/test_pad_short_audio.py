from __future__ import annotations

import numpy as np

from mteb.models.modality_collators import conv_receptive_field, pad_to_min_samples


def test_conv_receptive_field_wav2vec2_base():
    # https://huggingface.co/facebook/wav2vec2-base/blob/main/config.json
    assert conv_receptive_field((10, 3, 3, 3, 3, 2, 2), (5, 2, 2, 2, 2, 2, 2)) == 400


def test_pad_to_min_samples_pads_only_short_clips():
    short = np.ones(368, dtype=np.float32)
    padded = pad_to_min_samples(short, 400)
    assert padded.shape == (400,)
    assert np.array_equal(padded[:368], short)
    assert not padded[368:].any()

    long = np.ones(1000, dtype=np.float32)
    assert np.array_equal(pad_to_min_samples(long, 400), long)


def test_pad_to_min_samples_pads_time_axis_of_multichannel_audio():
    stereo = np.ones((2, 100), dtype=np.float32)
    assert pad_to_min_samples(stereo, 400).shape == (2, 400)


def test_conv_receptive_field_sew_d_squeeze():
    # asapp/sew-d-tiny-100k-ft-ls100h: squeeze_factor 2 needs one more conv stride (320) -> 720 samples
    kernels = (10, 3, 1, 3, 1, 3, 1, 3, 1, 2, 1, 2, 1)
    strides = (5, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1)
    assert conv_receptive_field(kernels, strides) == 400
    assert conv_receptive_field(kernels, strides, min_frames=2) == 720
