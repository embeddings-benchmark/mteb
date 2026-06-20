"""This implements minimal viable mock tasks for testing the benchmarking framework."""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from PIL.Image import Image

multilingual_eval_langs = {
    "eng": ["eng-Latn"],
    "fra": ["fra-Latn"],
}


def create_mock_images(np_rng: np.random.Generator, n: int = 2) -> list[Image]:
    from PIL import Image

    images = [np_rng.integers(0, 255, (100, 100, 3)) for _ in range(n)]
    return [Image.fromarray(image.astype("uint8")).convert("RGBA") for image in images]


def create_mock_video_bytes(
    np_rng: np.random.Generator,
    n: int = 2,
    width: int = 64,
    height: int = 64,
    fps: int = 24,
    duration_s: float = 1.0,
) -> list[bytes]:
    """Create minimal video bytes using PyAV.

    Args:
        np_rng: NumPy random generator used to produce distinct frame content per video.
        n: Number of video clips to generate.
        width: Frame width in pixels.
        height: Frame height in pixels.
        fps: Frames per second.
        duration_s: Duration of each video clip in seconds.

    Returns:
        List of n videos, each encoded as MP4 bytes.
    """
    try:
        import pytest

        pytest.importorskip("av", reason="Please, install av to run mock video tasks")
    except ImportError:
        try:
            import av
        except ImportError:
            raise ImportError("Please, install av to run mock video tasks")

    import av

    videos = []
    num_frames = int(fps * duration_s)
    for _ in range(n):
        buf = io.BytesIO()
        container = av.open(buf, mode="w", format="mp4")
        stream = container.add_stream("h264", rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        for _ in range(num_frames):
            frame_data = np_rng.integers(0, 255, (height, width, 3), dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(frame_data, format="rgb24")
            for pkt in stream.encode(frame):
                container.mux(pkt)
        for pkt in stream.encode():
            container.mux(pkt)
        container.close()
        videos.append(buf.getvalue())
    return videos


def create_mock_audio(
    np_rng: np.random.Generator,
    n: int = 2,
    duration_s: float = 1.0,
    sampling_rate: int = 16_000,
) -> list[np.ndarray]:
    audio_samples = []
    num_samples = int(duration_s * sampling_rate)
    for _ in range(n):
        audio = np_rng.uniform(-1.0, 1.0, num_samples).astype(np.float32)
        audio_samples.append(
            {
                "array": audio,
                "sampling_rate": sampling_rate,
            }
        )
    return audio_samples
