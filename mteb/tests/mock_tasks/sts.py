from __future__ import annotations

import datasets
from datasets import Audio, Dataset, DatasetDict

from mteb.abstasks.sts import AbsTaskSTS
from mteb.abstasks.task_metadata import TaskMetadata
from mteb.types import PromptType

from .utils import (
    create_mock_audio,
    create_mock_images,
    create_mock_video_bytes,
    general_args,
    multilingual_eval_langs,
)


class MockSTSTask(AbsTaskSTS):
    expected_stats = {
        "test": {
            "num_samples": 2,
            "number_of_characters": 113,
            "unique_pairs": 2,
            "text1_statistics": {
                "total_text_length": 52,
                "min_text_length": 23,
                "average_text_length": 26.0,
                "max_text_length": 29,
                "unique_texts": 2,
            },
            "text2_statistics": {
                "total_text_length": 61,
                "min_text_length": 24,
                "average_text_length": 30.5,
                "max_text_length": 37,
                "unique_texts": 2,
            },
            "image1_statistics": None,
            "image2_statistics": None,
            "audio1_statistics": None,
            "video1_statistics": None,
            "audio2_statistics": None,
            "video2_statistics": None,
            "label_statistics": {"min_score": 0, "avg_score": 0.5, "max_score": 1},
        }
    }

    metadata = TaskMetadata(
        type="STS",
        name="MockSTSTask",
        main_score="cosine_spearman",
        **general_args,
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        sentence1 = ["This is a test sentence", "This is another test sentence"]
        sentence2 = [
            "dette er en test sætning",
            "denne her matche ikke den ovenstående",
        ]  # "this is a test sentence", "this does not match the above"
        scores = [1, 0]

        self.dataset = DatasetDict(
            {
                "test": Dataset.from_dict(
                    {
                        "sentence1": sentence1,
                        "sentence2": sentence2,
                        "score": scores,
                    }
                ),
            }
        )
        self.data_loaded = True

    min_score = 0
    max_score = 1


class MockMultilingualSTSTask(AbsTaskSTS):
    expected_stats = {
        "test": {
            "num_samples": 4,
            "number_of_characters": 226,
            "unique_pairs": 2,
            "text1_statistics": {
                "total_text_length": 104,
                "min_text_length": 23,
                "average_text_length": 26.0,
                "max_text_length": 29,
                "unique_texts": 2,
            },
            "text2_statistics": {
                "total_text_length": 122,
                "min_text_length": 24,
                "average_text_length": 30.5,
                "max_text_length": 37,
                "unique_texts": 2,
            },
            "image1_statistics": None,
            "image2_statistics": None,
            "audio1_statistics": None,
            "video1_statistics": None,
            "audio2_statistics": None,
            "video2_statistics": None,
            "label_statistics": {"min_score": 0, "avg_score": 0.5, "max_score": 1},
            "hf_subset_descriptive_stats": {
                "eng": {
                    "num_samples": 2,
                    "number_of_characters": 113,
                    "unique_pairs": 2,
                    "text1_statistics": {
                        "total_text_length": 52,
                        "min_text_length": 23,
                        "average_text_length": 26.0,
                        "max_text_length": 29,
                        "unique_texts": 2,
                    },
                    "text2_statistics": {
                        "total_text_length": 61,
                        "min_text_length": 24,
                        "average_text_length": 30.5,
                        "max_text_length": 37,
                        "unique_texts": 2,
                    },
                    "image1_statistics": None,
                    "image2_statistics": None,
                    "audio1_statistics": None,
                    "video1_statistics": None,
                    "audio2_statistics": None,
                    "video2_statistics": None,
                    "label_statistics": {
                        "min_score": 0,
                        "avg_score": 0.5,
                        "max_score": 1,
                    },
                },
                "fra": {
                    "num_samples": 2,
                    "number_of_characters": 113,
                    "unique_pairs": 2,
                    "text1_statistics": {
                        "total_text_length": 52,
                        "min_text_length": 23,
                        "average_text_length": 26.0,
                        "max_text_length": 29,
                        "unique_texts": 2,
                    },
                    "text2_statistics": {
                        "total_text_length": 61,
                        "min_text_length": 24,
                        "average_text_length": 30.5,
                        "max_text_length": 37,
                        "unique_texts": 2,
                    },
                    "image1_statistics": None,
                    "image2_statistics": None,
                    "audio1_statistics": None,
                    "video1_statistics": None,
                    "audio2_statistics": None,
                    "video2_statistics": None,
                    "label_statistics": {
                        "min_score": 0,
                        "avg_score": 0.5,
                        "max_score": 1,
                    },
                },
            },
        }
    }

    metadata = TaskMetadata(
        type="STS",
        name="MockMultilingualSTSTask",
        main_score="cosine_spearman",
        **general_args,
    )
    metadata.eval_langs = multilingual_eval_langs

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        sentence1 = ["This is a test sentence", "This is another test sentence"]
        sentence2 = [
            "dette er en test sætning",
            "denne her matche ikke den ovenstående",
        ]  # "this is a test sentence", "this does not match the above"
        scores = [1, 0]
        data = {
            "test": Dataset.from_dict(
                {
                    "sentence1": sentence1,
                    "sentence2": sentence2,
                    "score": scores,
                }
            ),
        }

        self.dataset = {}
        for lang in self.hf_subsets:
            self.dataset[lang] = data

        self.data_loaded = True

    min_score = 0
    max_score = 1


class MockVisualSTSTask(AbsTaskSTS):
    expected_stats = {
        "test": {
            "num_samples": 2,
            "unique_pairs": 2,
            "number_of_characters": None,
            "text1_statistics": None,
            "text2_statistics": None,
            "image1_statistics": {
                "min_image_width": 100,
                "average_image_width": 100.0,
                "max_image_width": 100,
                "min_image_height": 100,
                "average_image_height": 100.0,
                "max_image_height": 100,
                "unique_images": 2,
            },
            "image2_statistics": {
                "min_image_width": 100,
                "average_image_width": 100.0,
                "max_image_width": 100,
                "min_image_height": 100,
                "average_image_height": 100.0,
                "max_image_height": 100,
                "unique_images": 2,
            },
            "audio1_statistics": None,
            "video1_statistics": None,
            "audio2_statistics": None,
            "video2_statistics": None,
            "label_statistics": {"min_score": 0.5, "avg_score": 0.5, "max_score": 0.5},
        }
    }

    metadata = TaskMetadata(
        type="VisualSTS(eng)",
        name="MockVisualSTS",
        main_score="cosine_spearman",
        **general_args,
    )
    metadata.modalities = ["image"]
    metadata.category = "i2i"

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        images = create_mock_images(self.np_rng)
        scores = [0.5, 0.5]

        self.dataset = DatasetDict(
            {
                "test": Dataset.from_dict(
                    {
                        "sentence1": images,
                        "sentence2": images,
                        "score": scores,
                    }
                ),
            }
        )
        self.dataset = self.dataset.cast_column("sentence1", datasets.Image())
        self.dataset = self.dataset.cast_column("sentence2", datasets.Image())
        self.data_loaded = True


class MockVideoAudioSTSTask(AbsTaskSTS):
    metadata = TaskMetadata(
        type="STS",
        name="MockVideoAudioSTS",
        main_score="cosine_spearman",
        **general_args,
    )
    metadata.modalities = ["video", "audio"]
    metadata.category = "va2va"
    column_names = (
        {"video1": "video", "audio1": "audio"},
        {"video2": "video", "audio2": "audio"},
    )

    expected_stats = {
        "test": {
            "num_samples": 2,
            "unique_pairs": 2,
            "number_of_characters": None,
            "text1_statistics": None,
            "text2_statistics": None,
            "image1_statistics": None,
            "image2_statistics": None,
            "audio1_statistics": {
                "total_duration_seconds": 2.0,
                "min_duration_seconds": 1.0,
                "average_duration_seconds": 1.0,
                "max_duration_seconds": 1.0,
                "unique_audios": 2,
                "average_sampling_rate": 16000.0,
                "sampling_rates": {16000: 2},
            },
            "audio2_statistics": {
                "total_duration_seconds": 2.0,
                "min_duration_seconds": 1.0,
                "average_duration_seconds": 1.0,
                "max_duration_seconds": 1.0,
                "unique_audios": 2,
                "average_sampling_rate": 16000.0,
                "sampling_rates": {16000: 2},
            },
            "video1_statistics": {
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
            "video2_statistics": {
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
            "label_statistics": {"min_score": 0.5, "avg_score": 0.5, "max_score": 0.5},
        }
    }

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        from datasets import Video

        mock_videos = create_mock_video_bytes(self.np_rng)
        mock_audio = create_mock_audio(self.np_rng)

        self.dataset = DatasetDict(
            {
                "test": Dataset.from_dict(
                    {
                        "video1": mock_videos,
                        "audio1": mock_audio,
                        "video2": mock_videos,
                        "audio2": mock_audio,
                        "score": [0.5, 0.5],
                    }
                ),
            }
        )
        self.dataset = self.dataset.cast_column("video1", Video())
        self.dataset = self.dataset.cast_column("video2", Video())
        self.dataset = self.dataset.cast_column("audio1", Audio())
        self.dataset = self.dataset.cast_column("audio2", Audio())
        self.data_loaded = True


class MockSymCustomVideoAudiSTSTask(AbsTaskSTS):
    """Asymmetric pair classification: side-1 is video only, side-2 is audio only. Differs from v1 by `input_column_name` is str instead of list"""

    metadata = TaskMetadata(
        type="VideoPairClassification",
        name="STS",
        main_score="cosine_spearman",
        **general_args,  # type: ignore[arg-type]
    )
    metadata.modalities = ["video", "audio"]
    metadata.category = "v2a"

    column_names = (
        {"video": "video"},
        {"audio": "audio"},
    )

    input1_prompt_type = PromptType.document
    input2_prompt_type = PromptType.document

    expected_stats = {
        "test": {
            "num_samples": 2,
            "number_of_characters": None,
            "unique_pairs": 2,
            "text1_statistics": None,
            "text2_statistics": None,
            "image1_statistics": None,
            "image2_statistics": None,
            "audio1_statistics": None,
            "audio2_statistics": {
                "total_duration_seconds": 2.0,
                "min_duration_seconds": 1.0,
                "average_duration_seconds": 1.0,
                "max_duration_seconds": 1.0,
                "unique_audios": 2,
                "average_sampling_rate": 16000.0,
                "sampling_rates": {16000: 2},
            },
            "video1_statistics": {
                "total_duration_seconds": 2,
                "total_frames": 48,
                "min_width": 64,
                "average_width": 64.0,
                "max_width": 64,
                "min_height": 64,
                "average_height": 64.0,
                "max_height": 64,
                "min_duration_seconds": 1,
                "average_duration_seconds": 1.0,
                "max_duration_seconds": 1,
                "unique_videos": 2,
                "average_fps": 24.0,
                "fps": {24: 2},
                "min_resolution": (64, 64),
                "average_resolution": (64.0, 64.0),
                "max_resolution": (64, 64),
                "resolutions": {"64x64": 2},
            },
            "video2_statistics": None,
            "label_statistics": {"min_score": 0, "avg_score": 0.5, "max_score": 1},
        }
    }

    def load_data(self, **kwargs):
        from datasets import Video

        mock_videos = create_mock_video_bytes(self.np_rng)
        mock_audio = create_mock_audio(self.np_rng)

        self.dataset = DatasetDict(
            {
                "test": Dataset.from_dict(
                    {
                        "video": mock_videos,
                        "audio": mock_audio,
                        "score": [0, 1],
                    }
                ),
            }
        )
        self.dataset = self.dataset.cast_column("video", Video())
        self.dataset = self.dataset.cast_column("audio", Audio())
        self.data_loaded = True
