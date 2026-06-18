"""This implements minimal viable mock tasks for testing the benchmarking framework."""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

import numpy as np
import pytest
from datasets import Dataset

from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData

if TYPE_CHECKING:
    from PIL.Image import Image

general_args = {
    "description": "a mock task for testing",
    "reference": "https://github.com/embeddings-benchmark/mteb",
    "dataset": {
        "path": "NA",
        "revision": "NA",
    },
    "category": "t2t",
    "eval_splits": ["test"],
    "eval_langs": ["eng-Latn"],
    "date": ("2022-12-22", "2022-12-22"),
    "dialect": ["Written"],
    "domains": [],
    "task_subtypes": [],
    "license": "cc-by-4.0",
    "annotations_creators": "derived",
    "modalities": ["text"],
    "sample_creation": "found",
    "bibtex_citation": "",
}

multilingual_eval_langs = {
    "eng": ["eng-Latn"],
    "fra": ["fra-Latn"],
}


def base_retrieval_datasplit() -> RetrievalSplitData:
    return RetrievalSplitData(
        queries=Dataset.from_list(
            [
                {
                    "id": "q1",
                    "text": "This is a test sentence",
                },
                {
                    "id": "q2",
                    "text": "This is another test sentence",
                },
            ]
        ),
        corpus=Dataset.from_list(
            [
                {
                    "id": "d2",
                    "text": "This is a positive sentence",
                    "title": "Title of d1",
                },
                {
                    "id": "d1",
                    "text": "This is another positive sentence",
                    "title": "Title of d2",
                },
            ]
        ),
        relevant_docs={
            "q1": {"d1": 1, "d2": 0},
            "q2": {"d1": 0, "d2": 1},
        },
        top_ranked={
            "q1": ["d1", "d2"],
            "q2": ["d2", "d1"],
        },
    )


def instruction_retrieval_datasplit() -> RetrievalSplitData:
    base_ds = base_retrieval_datasplit()
    base_ds["queries"] = Dataset.from_list(
        [
            {
                "id": "q1",
                "text": "This is a test sentence",
                "instruction": "This is a test instruction",
            },
            {
                "id": "q2",
                "text": "This is another test sentence",
                "instruction": "This is another test instruction",
            },
        ]
    )
    return base_ds


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
    pytest.importorskip("av", reason="Please, install av to run mock video tasks")

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


_VIDEO_TEXTS = [
    "This is a video of an action",
    "This is another video of a scene",
]
