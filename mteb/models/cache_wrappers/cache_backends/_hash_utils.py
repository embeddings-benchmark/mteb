from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

    import numpy as np
    from PIL import Image


def _hash_item(item: Mapping[str, Any]) -> str:
    item_hash = ""
    if "text" in item:
        item_text: str = item["text"]
        item_hash = hashlib.sha256(item_text.encode()).hexdigest()

    if "image" in item:
        image: Image.Image = item["image"]
        item_hash += hashlib.sha256(image.tobytes()).hexdigest()

    if "audio" in item:
        audio = item["audio"]
        audio_array: np.ndarray = audio["array"]
        sampling_rate: int = audio["sampling_rate"]
        audio_bytes = audio_array.tobytes() + str(sampling_rate).encode()
        item_hash += hashlib.sha256(audio_bytes).hexdigest()

    if "video" in item:
        video = item["video"]
        # Hash the first decoded frame as a stable fingerprint for the video.
        # VideoDecoder.get_frames_at returns a FrameBatch; .data is a NCHW/NHWC tensor.
        try:
            import torch

            frame_tensor = video.get_frames_at([0]).data
            if isinstance(frame_tensor, torch.Tensor):
                frame_bytes = frame_tensor.cpu().numpy().tobytes()
            else:
                frame_bytes = frame_tensor.tobytes()
        except Exception:
            # Fall back to metadata-based hash (num_frames + duration) if decoding fails
            meta = video.metadata
            fallback = f"{meta.num_frames}:{meta.end_stream_seconds}"
            frame_bytes = fallback.encode()
        item_hash += hashlib.sha256(frame_bytes).hexdigest()

    if len(item_hash) == 0:
        raise TypeError(f"Unsupported cache key type: {type(item)}")

    return item_hash
