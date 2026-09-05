"""Per-sample content hashes, used to tell whether two samples hold the same content.

Shared by the descriptive statistics and by the filters in `mteb.quality`, so that both agree on what makes two
images, audio clips or videos identical.
"""

from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

from tqdm.auto import tqdm

if TYPE_CHECKING:
    from PIL import Image
    from torchcodec.decoders import VideoDecoder  # type: ignore[attr-defined]

    from mteb.types import Modalities
    from mteb.types._encoder_io import AudioInputItem


def compute_text_hashes(texts: list[str], max_workers: int | None = None) -> list[str]:
    """Return a hash per text — for text, the string itself is the identity key."""
    return texts


def compute_image_hashes(
    images: list[Image.Image], max_workers: int | None = None
) -> list[str]:
    """Return a per-image MD5 hash of the raw pixel bytes."""

    def _hash_one(img: Image.Image) -> str:
        return hashlib.md5(img.tobytes(), usedforsecurity=False).hexdigest()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(_hash_one, images))


def compute_audio_hashes(
    audios: list[AudioInputItem], max_workers: int | None = None
) -> list[str]:
    """Return a per-audio MD5 hash of the raw sample array bytes."""

    def _hash_one(audio: AudioInputItem) -> str:
        return hashlib.md5(audio["array"].tobytes(), usedforsecurity=False).hexdigest()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(_hash_one, audios))


def compute_video_hashes(
    videos: list[VideoDecoder], max_workers: int | None = None
) -> list[str]:
    """Return a per-video MD5 hash derived from the first decoded frame.

    Decoding a frame is the most expensive part of video statistics; this function
    is extracted so callers can pass the resulting list to ``calculate_video_statistics``
    and avoid repeating the decode.
    """

    def _hash_one(video: VideoDecoder) -> str:
        meta = video.metadata
        # Drop the last frame index because some container metadata over-counts
        # by one (the final claimed frame fails to decode).
        num_frames = meta.num_frames - 1 if meta.num_frames else meta.num_frames
        avg_fps = meta.average_fps

        if num_frames is None or num_frames == 0:
            raise ValueError(f"Number of frames is {num_frames}")

        if num_frames is not None and avg_fps is not None and avg_fps > 0:
            step = max(1, round(avg_fps))
            frame_indices = list(range(0, num_frames, step))
        else:
            frame_indices = [0]

        frames = video.get_frames_at(frame_indices).data
        return hashlib.md5(frames.numpy().tobytes(), usedforsecurity=False).hexdigest()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = list(
            tqdm(
                executor.map(_hash_one, videos),
                total=len(videos),
                desc="Computing video hashes",
            )
        )
    return futures


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
