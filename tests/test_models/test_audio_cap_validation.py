"""Wrappers that pass the audio cap to the feature extractor, not AudioCollator,
must reject a non-positive cap themselves, before loading weights.
"""

from __future__ import annotations

import importlib
import inspect

import pytest

CASES = [
    ("audio_qwen2_models", "Qwen2AudioWrapper", "max_audio_length_seconds"),
    ("mctct_model", "MCTCTWrapper", "max_audio_length_seconds"),
    ("sewd_models", "SewDWrapper", "max_audio_length_seconds"),
    ("speecht5_models", "SpeechT5Audio", "max_audio_length_seconds"),
    ("wavlm_models", "WavlmWrapper", "max_audio_length_seconds"),
]


@pytest.mark.parametrize(
    ("module", "cls_name", "cap_kw"), CASES, ids=[c[1] for c in CASES]
)
@pytest.mark.parametrize("cap", [0, -1])
def test_rejects_non_positive_cap(
    module: str, cls_name: str, cap_kw: str, cap: int
) -> None:
    mod = importlib.import_module(f"mteb.models.model_implementations.{module}")
    cls = getattr(mod, cls_name)
    params = inspect.signature(cls.__init__).parameters
    assert cap_kw in params, f"{cls_name} has no {cap_kw}"
    first = next(p for p in params if p != "self")
    kwargs = {first: "not-a-real-checkpoint", cap_kw: cap}
    if "revision" in params:
        kwargs["revision"] = "unused"
    with pytest.raises(ValueError, match="must be positive"):
        cls(**kwargs)
