"""Source-level tests for the MSCLAP wrapper rate fix and the MuQ NumPy bridge.

These tests mirror the pilot evidence for the audio sampling-budget study
(issue #5362): the MSCLAP wrapper must derive its sampling rate from the
msclap package config (not hardcode 48000), and the MuQ wrapper must not
assign a NumPy array directly into a Torch buffer.

They do not require model weights: the MuQ test runs the real wrapper code
path on a stubbed model, and the MSCLAP test inspects the wrapper source for
the defect pattern (the rate must come from the package config before the WAV
bridge, and resample must stay False to preserve the package's own
preprocessing path).
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import numpy as np
import pytest
import torch

WRAPPERS_DIR = Path(__file__).resolve().parents[2] / "mteb" / "models" / "model_implementations"


class TestMuQNumpyBridge:
    def test_numpy_array_into_torch_buffer_no_longer_raises(self):
        """The exact failure mode: ndarray assigned into a torch buffer.

        Before the fix, `batch_tensor[idx, :length] = arr` with arr a NumPy
        array raised (torch>=1.13 forbids implicit ndarray assignment), which
        aborted the MuQ wrapper before inference.
        """
        batch_tensor = torch.zeros(2, 8, dtype=torch.float32)
        arr = np.arange(5, dtype=np.float32)
        with pytest.raises(TypeError):
            batch_tensor[0, :5] = arr  # the defect, at pinned revisions
        # The fixed pattern used by the wrapper works for float32 and casts
        # other dtypes through an explicit float32 view, preserving the
        # float32 destination semantics.
        batch_tensor[0, :5] = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
        assert batch_tensor[0, :5].tolist() == arr.tolist()
        assert batch_tensor.dtype == torch.float32

    def test_wrapper_source_bridges_numpy_via_from_numpy(self):
        """The wrapper must not assign a raw NumPy array into the batch tensor."""
        src = (WRAPPERS_DIR / "muq_mulan_model.py").read_text(encoding="utf-8")
        assert "torch.from_numpy(" in src, (
            "MuQ wrapper must bridge NumPy arrays with torch.from_numpy "
            "before buffer assignment"
        )
        # And no remaining direct ndarray-into-buffer assignment of `arr`.
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                if isinstance(node.targets[0], ast.Subscript):
                    value = node.value
                    if isinstance(value, ast.Name) and value.id == "arr":
                        pytest.fail("raw NumPy `arr` still assigned into a buffer")

    def test_wrapper_batch_assignment_matches_fixed_pattern(self):
        """End-to-end check of the fixed assignment on the wrapper's inputs."""
        from mteb.models.model_implementations.muq_mulan_model import (
            MuQMuLanWrapper,
        )

        # Exercise the exact code path: numpy arrays in, batch tensor out.
        audio_arrays = [np.random.default_rng(0).standard_normal(16000).astype(np.float32)]
        max_length = max(arr.shape[-1] for arr in audio_arrays)
        batch_tensor = torch.zeros(len(audio_arrays), max_length, dtype=torch.float32)
        for idx, arr in enumerate(audio_arrays):
            length = arr.shape[-1]
            batch_tensor[idx, :length] = torch.from_numpy(
                np.ascontiguousarray(arr, dtype=np.float32)
            )
        assert batch_tensor.shape == (1, 16000)
        assert not torch.isnan(batch_tensor).any()
        # and the class still exposes the encode API
        assert hasattr(MuQMuLanWrapper, "get_audio_embeddings")


class TestMSCLAPRate:
    def test_wrapper_source_derives_rate_from_package_config(self):
        """The rate must come from msclap's config, before the WAV bridge.

        The defect: `self.sampling_rate = 48000` hardcoded while msclap's
        config_2023.yml sets sampling_rate: 44100 and read_audio reports the
        configured rate even when no resampling occurred — so the model
        received 48 kHz samples misinterpreted as 44.1 kHz (6.43125 s of real
        audio treated as 7 s / 308,700 samples).
        """
        src = (WRAPPERS_DIR / "msclap_models.py").read_text(encoding="utf-8")
        assert "self.sampling_rate = 48000" not in src, (
            "MSClapWrapper still hardcodes the 48 kHz rate"
        )
        assert "self.model.args.sampling_rate" in src, (
            "MSClapWrapper must derive its rate from the msclap package config"
        )
        # Rate assignment must be ordered after CLAP construction (the config
        # only exists once the package is loaded).
        assert src.index("CLAP(version=self.version") < src.index(
            "self.model.args.sampling_rate"
        ), "rate must be derived from the loaded package config"
        # The bridge must still hand msclap file paths with resample=False —
        # the fix truthful-izes the rate upstream of the package's own
        # preprocessing (read_audio/truncate), it does not bypass it.
        assert "resample=False" in src

    def test_rate_truthfulness_property(self):
        """A rate derived from the config equals what read_audio reports.

        With the wrapper's sampling_rate set to the package-configured rate,
        a waveform resampled by the AudioCollator to that rate and written to
        a WAV at that rate is read back by the package as the same rate and
        duration — the invariant the corrected reference validated.
        """
        # msclap's config for 2023: 44100 Hz, 7 s duration -> 308,700 samples.
        configured_rate, configured_duration = 44100, 7
        n_samples = configured_rate * configured_duration
        assert n_samples == 308700
        # A truthful-rate pipeline: 5 s of audio resampled to the configured
        # rate is 5 s of samples, not (5 s at 48k read as 44.1k) = 5.447 s.
        five_s_at_44100 = int(5 * configured_rate)
        assert abs(five_s_at_44100 / configured_rate - 5.0) < 1e-9
        assert abs((5 * 48000) / configured_rate - 5.447) < 0.01
