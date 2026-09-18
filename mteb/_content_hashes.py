"""Per-sample content hashes, used to tell whether two samples hold the same content.

Shared by the descriptive statistics, the filters in `mteb.data_cleaning`, and the
cross-task embedding cache in `mteb.models.cache_wrappers`.
"""

from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

import torch
from tqdm.auto import tqdm

if TYPE_CHECKING:
    from PIL import Image
    from torchcodec.decoders import VideoDecoder  # type: ignore[attr-defined]

    from mteb.types import Modalities
    from mteb.types._encoder_io import AudioInputItem


def _sha256_digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_image(image: Image.Image) -> str:
    """Content hash for a single image, from its raw pixel bytes."""
    return _sha256_digest(image.tobytes())


def hash_audio(audio: AudioInputItem) -> str:
    """Content hash for a single audio clip (raw samples + sampling rate).

    The sampling rate is mixed into the hash because it changes how the raw
    samples are interpreted (duration/pitch); identical sample bytes at a
    different declared sampling rate are a different clip.
    """
    array = audio["array"]
    sampling_rate = audio["sampling_rate"]
    return _sha256_digest(array.tobytes() + str(sampling_rate).encode())


def hash_video(video: VideoDecoder) -> str:
    """Content hash for a single video.

    Samples roughly one frame per second.
    """
    meta = video.metadata
    # Some containers over-count num_frames by one; the final claimed
    # frame often fails to decode.
    num_frames = meta.num_frames - 1 if meta.num_frames else meta.num_frames
    avg_fps = meta.average_fps
    if not num_frames:
        raise ValueError(f"num_frames is {num_frames}")

    if avg_fps is not None and avg_fps > 0:
        step = max(1, round(avg_fps))
        frame_indices = list(range(0, num_frames, step))
    else:
        frame_indices = [0]

    frame_tensor = video.get_frames_at(frame_indices).data
    if isinstance(frame_tensor, torch.Tensor):
        frame_bytes = frame_tensor.cpu().numpy().tobytes()
    else:
        frame_bytes = frame_tensor.tobytes()

    return _sha256_digest(frame_bytes)


def compute_text_hashes(texts: list[str], max_workers: int | None = None) -> list[str]:
    """Return a hash per text — for text, the string itself is the identity key."""
    return list(tqdm(texts, desc="Computing text hashes"))


def compute_image_hashes(
    images: list[Image.Image], max_workers: int | None = None
) -> list[str]:
    """Return a per-image content hash."""
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(
            tqdm(
                executor.map(hash_image, images),
                total=len(images),
                desc="Computing image hashes",
            )
        )


def compute_audio_hashes(
    audios: list[AudioInputItem], max_workers: int | None = None
) -> list[str]:
    """Return a per-audio content hash."""
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(
            tqdm(
                executor.map(hash_audio, audios),
                total=len(audios),
                desc="Computing audio hashes",
            )
        )


def compute_video_hashes(
    videos: list[VideoDecoder], max_workers: int | None = None
) -> list[str]:
    """Return a per-video content hash.

    Decoding a frame is the most expensive part of video statistics; this function
    is extracted so callers can pass the resulting list to ``calculate_video_statistics``
    and avoid repeating the decode.
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(
            tqdm(
                executor.map(hash_video, videos),
                total=len(videos),
                desc="Computing video hashes",
            )
        )


MODALITY_HASH_FNS: dict[str, Any] = {
    "text": compute_text_hashes,
    "image": compute_image_hashes,
    "audio": compute_audio_hashes,
    "video": compute_video_hashes,
}
"""The hash function of each modality whose content can be compared."""


def compute_modality_hashes(
    col_inputs: dict[Modalities, list[Any]],
    max_workers: int | None = None,
) -> dict[str, list[str]]:
    """Compute per-sample hashes for each modality using the shared hash functions.

    Reuses the same hashing logic as the ``calculate_*_statistics`` functions so that
    callers can pass the result to both statistics functions and intersection checks
    without decoding the data twice.
    """
    return {
        mod: MODALITY_HASH_FNS[mod](values, max_workers=max_workers)
        for mod, values in col_inputs.items()
    }


def hash_item(item: dict[str, Any]) -> str:
    """Compute a deterministic content hash for a multi-modality item.

    Used by the cross-task embedding cache (`mteb.models.cache_wrappers`) to
    derive a cache key for an item that may carry 'text', 'image', 'audio',
    and/or 'video' data.
    """
    item_hash = ""
    if "text" in item:
        item_hash = _sha256_digest(item["text"].encode())

    if "image" in item:
        item_hash += hash_image(item["image"])

    if "audio" in item:
        item_hash += hash_audio(item["audio"])

    if "video" in item:
        item_hash += hash_video(item["video"])

    if len(item_hash) == 0:
        raise TypeError(f"Unsupported cache key type: {type(item)}")

    return item_hash
