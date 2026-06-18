from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from datasets import Audio, Dataset, DatasetDict

from mteb.abstasks.task_metadata import TaskMetadata
from mteb.abstasks.zeroshot_classification import (
    AbsTaskZeroShotClassification,
)

if TYPE_CHECKING:
    pass

from .utils import (
    create_mock_audio,
    create_mock_images,
    create_mock_video_bytes,
    general_args,
)


class MockZeroShotClassificationTask(AbsTaskZeroShotClassification):
    expected_stats = {
        "test": {
            "num_samples": 2,
            "text_statistics": None,
            "image_statistics": {
                "min_image_width": 100,
                "average_image_width": 100.0,
                "max_image_width": 100,
                "min_image_height": 100,
                "average_image_height": 100.0,
                "max_image_height": 100,
                "unique_images": 2,
            },
            "audio_statistics": None,
            "video_statistics": None,
            "label_statistics": {
                "min_labels_per_text": 1,
                "average_label_per_text": 1.0,
                "max_labels_per_text": 1,
                "unique_labels": 2,
                "labels": {"0": {"count": 1}, "1": {"count": 1}},
            },
            "candidates_labels_text_statistics": {
                "total_text_length": 52,
                "min_text_length": 23,
                "average_text_length": 26.0,
                "max_text_length": 29,
                "unique_texts": 2,
            },
        }
    }

    metadata = TaskMetadata(
        type="ZeroShotClassification",
        name="MockZeroShotClassification",
        main_score="accuracy",
        **general_args,
    )
    metadata.modalities = ["image", "text"]
    metadata.category = "i2t"

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        images = create_mock_images(self.np_rng)
        labels = [0, 1]

        self.dataset = DatasetDict(
            {
                "test": Dataset.from_dict(
                    {
                        "image": images,
                        "label": labels,
                    }
                ),
            }
        )
        self.data_loaded = True

    def get_candidate_labels(self) -> list[str]:  # noqa: PLR6301
        return ["This is a test sentence", "This is another test sentence"]


class MockTextZeroShotClassificationTask(AbsTaskZeroShotClassification):
    expected_stats = {
        "test": {
            "num_samples": 2,
            "text_statistics": {
                "total_text_length": 52,
                "min_text_length": 23,
                "average_text_length": 26.0,
                "max_text_length": 29,
                "unique_texts": 2,
            },
            "image_statistics": None,
            "audio_statistics": None,
            "video_statistics": None,
            "label_statistics": {
                "min_labels_per_text": 1,
                "average_label_per_text": 1.0,
                "max_labels_per_text": 1,
                "unique_labels": 2,
                "labels": {
                    "This is a test sentence": {"count": 1},
                    "This is another test sentence": {"count": 1},
                },
            },
            "candidates_labels_text_statistics": {
                "total_text_length": 52,
                "min_text_length": 23,
                "average_text_length": 26.0,
                "max_text_length": 29,
                "unique_texts": 2,
            },
        }
    }

    metadata = TaskMetadata(
        type="ZeroShotClassification",
        name="MockTextZeroShotClassification",
        main_score="accuracy",
        **general_args,
    )
    metadata.modalities = ["text"]
    metadata.category = "t2t"
    input_column_name = "text"

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        texts = ["This is a test sentence", "This is another test sentence"]
        # String labels matching `get_candidate_labels` cover the string-label
        # code path; they are mapped to candidate indices during evaluation.
        labels = ["This is a test sentence", "This is another test sentence"]

        self.dataset = DatasetDict(
            {
                "test": Dataset.from_dict(
                    {
                        "text": texts,
                        "label": labels,
                    }
                ),
            }
        )
        self.data_loaded = True

    def get_candidate_labels(self) -> list[str]:  # noqa: PLR6301
        return ["This is a test sentence", "This is another test sentence"]


class MockAudioZeroshotClassificationTask(AbsTaskZeroShotClassification):
    input_column_name: str = "audio"
    label_column_name: str = "label"

    expected_stats = {
        "test": {
            "num_samples": 2,
            "text_statistics": None,
            "image_statistics": None,
            "audio_statistics": {
                "total_duration_seconds": 2.0,
                "min_duration_seconds": 1.0,
                "average_duration_seconds": 1.0,
                "max_duration_seconds": 1.0,
                "unique_audios": 2,
                "average_sampling_rate": 16000.0,
                "sampling_rates": {16000: 2},
            },
            "video_statistics": None,
            "label_statistics": {
                "min_labels_per_text": 1,
                "average_label_per_text": 1.0,
                "max_labels_per_text": 1,
                "unique_labels": 2,
                "labels": {"0": {"count": 1}, "1": {"count": 1}},
            },
            "candidates_labels_text_statistics": {
                "total_text_length": 40,
                "min_text_length": 20,
                "average_text_length": 20.0,
                "max_text_length": 20,
                "unique_texts": 2,
            },
        }
    }

    metadata = TaskMetadata(
        type="AudioZeroshotClassification",
        name="MockAudioZeroshotClassification",
        main_score="accuracy",
        **general_args,
    )
    metadata.modalities = ["audio"]

    def load_data(self, **kwargs):
        mock_audio = create_mock_audio(self.np_rng)
        labels = np.array([0, 1])  # Convert labels to numpy array

        self.dataset = DatasetDict(
            {
                "test": Dataset.from_dict(
                    {
                        "audio": mock_audio,
                        "label": labels,
                    }
                ),
            }
        )
        self.dataset = self.dataset.cast_column("audio", Audio())
        self.data_loaded = True

    def get_candidate_labels(self) -> list[str]:  # noqa: PLR6301
        """Return the text candidates for zeroshot classification"""
        return ["This is sound type 0", "This is sound type 1"]


class MockVideoZeroshotClassificationTask(AbsTaskZeroShotClassification):
    input_column_name: str = "video"
    label_column_name: str = "label"

    expected_stats = {
        "test": {
            "num_samples": 2,
            "text_statistics": None,
            "image_statistics": None,
            "audio_statistics": None,
            "video_statistics": {
                "total_duration_seconds": 2.0,
                "total_frames": 48,
                "min_width": 64,
                "average_width": 64.0,
                "max_width": 64,
                "min_height": 64,
                "average_height": 64.0,
                "max_height": 64,
                "min_duration_seconds": 1.0,
                "average_duration_seconds": 1.0,
                "max_duration_seconds": 1.0,
                "unique_videos": 2,
                "average_fps": 24.0,
                "fps": {24: 2},
                "min_resolution": (64, 64),
                "average_resolution": (64.0, 64.0),
                "max_resolution": (64, 64),
                "resolutions": {"64x64": 2},
            },
            "label_statistics": {
                "min_labels_per_text": 1,
                "average_label_per_text": 1.0,
                "max_labels_per_text": 1,
                "unique_labels": 2,
                "labels": {"0": {"count": 1}, "1": {"count": 1}},
            },
            "candidates_labels_text_statistics": {
                "total_text_length": 40,
                "min_text_length": 20,
                "average_text_length": 20.0,
                "max_text_length": 20,
                "unique_texts": 2,
            },
        }
    }

    metadata = TaskMetadata(
        type="VideoZeroshotClassification",
        name="MockVideoZeroshotClassification",
        main_score="accuracy",
        **general_args,
    )
    metadata.modalities = ["video"]
    metadata.category = "v2c"

    def load_data(self, **kwargs):
        from datasets import Video

        mock_videos = create_mock_video_bytes(self.np_rng)
        labels = np.array([0, 1])

        self.dataset = DatasetDict(
            {
                "test": Dataset.from_dict(
                    {
                        "video": mock_videos,
                        "label": labels,
                    }
                ),
            }
        )
        self.dataset = self.dataset.cast_column("video", Video())
        self.data_loaded = True

    def get_candidate_labels(self) -> list[str]:  # noqa: PLR6301
        return ["This is video type 0", "This is video type 1"]


class MockVideoAudioZeroshotClassificationTask(AbsTaskZeroShotClassification):
    input_column_name = ("video", "audio")
    label_column_name = "label"

    expected_stats = {
        "test": {
            "num_samples": 2,
            "text_statistics": None,
            "image_statistics": None,
            "audio_statistics": {
                "total_duration_seconds": 2.0,
                "min_duration_seconds": 1.0,
                "average_duration_seconds": 1.0,
                "max_duration_seconds": 1.0,
                "unique_audios": 2,
                "average_sampling_rate": 16000.0,
                "sampling_rates": {16000: 2},
            },
            "video_statistics": {
                "total_duration_seconds": 2.0,
                "total_frames": 48,
                "min_width": 64,
                "average_width": 64.0,
                "max_width": 64,
                "min_height": 64,
                "average_height": 64.0,
                "max_height": 64,
                "min_duration_seconds": 1.0,
                "average_duration_seconds": 1.0,
                "max_duration_seconds": 1.0,
                "unique_videos": 2,
                "average_fps": 24.0,
                "fps": {24: 2},
                "min_resolution": (64, 64),
                "average_resolution": (64.0, 64.0),
                "max_resolution": (64, 64),
                "resolutions": {"64x64": 2},
            },
            "label_statistics": {
                "min_labels_per_text": 1,
                "average_label_per_text": 1.0,
                "max_labels_per_text": 1,
                "unique_labels": 2,
                "labels": {"0": {"count": 1}, "1": {"count": 1}},
            },
            "candidates_labels_text_statistics": {
                "total_text_length": 52,
                "min_text_length": 26,
                "average_text_length": 26.0,
                "max_text_length": 26,
                "unique_texts": 2,
            },
        }
    }

    metadata = TaskMetadata(
        type="VideoZeroshotClassification",
        name="MockVideoZeroshotClassification",
        main_score="accuracy",
        **general_args,
    )
    metadata.modalities = ["video"]
    metadata.category = "v2c"

    def load_data(self, **kwargs):
        from datasets import Video

        mock_videos = create_mock_video_bytes(self.np_rng)
        mock_audio = create_mock_audio(self.np_rng)

        labels = np.array([0, 1])

        self.dataset = DatasetDict(
            {
                "test": Dataset.from_dict(
                    {
                        "video": mock_videos,
                        "audio": mock_audio,
                        "label": labels,
                    }
                ),
            }
        )
        self.dataset = self.dataset.cast_column("video", Video())
        self.dataset = self.dataset.cast_column("audio", Audio())
        self.data_loaded = True

    def get_candidate_labels(self) -> list[str]:  # noqa: PLR6301
        return ["This is video audio type 0", "This is video audio type 1"]
