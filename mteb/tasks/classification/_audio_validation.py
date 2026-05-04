from __future__ import annotations

from typing import Any

import numpy as np
import torch


def _audio_array_from_example(
    example: dict[str, Any],
    *,
    audio_key: str = "audio",
) -> Any | None:
    audio = example.get(audio_key)
    if audio is None:
        return None

    if isinstance(audio, dict):
        return audio.get("array")

    if hasattr(audio, "get_all_samples"):
        samples = audio.get_all_samples()
        return getattr(samples, "data", None)

    return getattr(audio, "array", None)


def is_valid_audio_example(
    example: dict[str, Any],
    *,
    audio_key: str = "audio",
    min_samples: int = 500,
) -> bool:
    audio_arr = _audio_array_from_example(example, audio_key=audio_key)
    if audio_arr is None:
        return False

    if isinstance(audio_arr, torch.Tensor):
        flat = audio_arr.detach().float().reshape(-1)
        if flat.numel() < min_samples:
            return False
        return bool(torch.isfinite(flat).all().item())

    flat_np = np.asarray(audio_arr, dtype=np.float32).reshape(-1)
    if flat_np.size < min_samples:
        return False
    return bool(np.isfinite(flat_np).all())
