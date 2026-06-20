from __future__ import annotations

from datasets import Audio, Dataset, DatasetDict

from mteb.abstasks.image.image_text_pair_classification import (
    AbsTaskImageTextPairClassification,
)
from mteb.abstasks.pair_classification import AbsTaskPairClassification
from mteb.abstasks.task_metadata import TaskMetadata
from mteb.types import PromptType

from .create_mock_samples import (
    create_mock_audio,
    create_mock_images,
    create_mock_video_bytes,
    multilingual_eval_langs,
)

general_args = {
    "description": "A lightweight mock pair classification task designed for testing, debugging, and local model verification within the MTEB framework.",
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


class MockPairClassificationTask(AbsTaskPairClassification):
    expected_stats = {
        "test": {
            "num_samples": 2,
            "unique_pairs": 2,
            "number_of_characters": 113,
            "text1_statistics": {
                "total_text_length": 52,
                "min_text_length": 23,
                "average_text_length": 26.0,
                "max_text_length": 29,
                "unique_texts": 2,
            },
            "image1_statistics": None,
            "audio1_statistics": None,
            "video1_statistics": None,
            "text2_statistics": {
                "total_text_length": 61,
                "min_text_length": 24,
                "average_text_length": 30.5,
                "max_text_length": 37,
                "unique_texts": 2,
            },
            "image2_statistics": None,
            "audio2_statistics": None,
            "video2_statistics": None,
            "labels_statistics": {
                "min_labels_per_text": 1,
                "average_label_per_text": 1.0,
                "max_labels_per_text": 1,
                "unique_labels": 2,
                "labels": {"1": {"count": 1}, "0": {"count": 1}},
            },
        }
    }

    metadata = TaskMetadata(
        type="PairClassification",
        name="MockPairClassificationTask",
        main_score="similarity_ap",
        **general_args,
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        sentence1 = ["This is a test sentence", "This is another test sentence"]
        sentence2 = [
            "dette er en test sætning",
            "denne her matche ikke den ovenstående",
        ]
        labels = [1, 0]

        self.dataset = DatasetDict(
            {
                "test": Dataset.from_dict(
                    {
                        "sentence1": sentence1,
                        "sentence2": sentence2,
                        "labels": labels,
                    }
                ),
            }
        )
        self.data_loaded = True


class MockMultilingualPairClassificationTask(AbsTaskPairClassification):
    expected_stats = {
        "test": {
            "num_samples": 4,
            "unique_pairs": 2,
            "number_of_characters": 226,
            "text1_statistics": {
                "total_text_length": 104,
                "min_text_length": 23,
                "average_text_length": 26.0,
                "max_text_length": 29,
                "unique_texts": 2,
            },
            "image1_statistics": None,
            "audio1_statistics": None,
            "video1_statistics": None,
            "text2_statistics": {
                "total_text_length": 122,
                "min_text_length": 24,
                "average_text_length": 30.5,
                "max_text_length": 37,
                "unique_texts": 2,
            },
            "image2_statistics": None,
            "audio2_statistics": None,
            "video2_statistics": None,
            "labels_statistics": {
                "min_labels_per_text": 1,
                "average_label_per_text": 1.0,
                "max_labels_per_text": 1,
                "unique_labels": 2,
                "labels": {"1": {"count": 2}, "0": {"count": 2}},
            },
            "hf_subset_descriptive_stats": {
                "eng": {
                    "num_samples": 2,
                    "unique_pairs": 2,
                    "number_of_characters": 113,
                    "text1_statistics": {
                        "total_text_length": 52,
                        "min_text_length": 23,
                        "average_text_length": 26.0,
                        "max_text_length": 29,
                        "unique_texts": 2,
                    },
                    "image1_statistics": None,
                    "audio1_statistics": None,
                    "video1_statistics": None,
                    "text2_statistics": {
                        "total_text_length": 61,
                        "min_text_length": 24,
                        "average_text_length": 30.5,
                        "max_text_length": 37,
                        "unique_texts": 2,
                    },
                    "image2_statistics": None,
                    "audio2_statistics": None,
                    "video2_statistics": None,
                    "labels_statistics": {
                        "min_labels_per_text": 1,
                        "average_label_per_text": 1.0,
                        "max_labels_per_text": 1,
                        "unique_labels": 2,
                        "labels": {"1": {"count": 1}, "0": {"count": 1}},
                    },
                },
                "fra": {
                    "num_samples": 2,
                    "unique_pairs": 2,
                    "number_of_characters": 113,
                    "text1_statistics": {
                        "total_text_length": 52,
                        "min_text_length": 23,
                        "average_text_length": 26.0,
                        "max_text_length": 29,
                        "unique_texts": 2,
                    },
                    "image1_statistics": None,
                    "audio1_statistics": None,
                    "video1_statistics": None,
                    "text2_statistics": {
                        "total_text_length": 61,
                        "min_text_length": 24,
                        "average_text_length": 30.5,
                        "max_text_length": 37,
                        "unique_texts": 2,
                    },
                    "image2_statistics": None,
                    "audio2_statistics": None,
                    "video2_statistics": None,
                    "labels_statistics": {
                        "min_labels_per_text": 1,
                        "average_label_per_text": 1.0,
                        "max_labels_per_text": 1,
                        "unique_labels": 2,
                        "labels": {"1": {"count": 1}, "0": {"count": 1}},
                    },
                },
            },
        }
    }

    metadata = TaskMetadata(
        type="PairClassification",
        name="MockMultilingualPairClassificationTask",
        main_score="similarity_ap",
        **general_args,
    )
    metadata.eval_langs = multilingual_eval_langs

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        sentence1 = ["This is a test sentence", "This is another test sentence"]
        sentence2 = [
            "dette er en test sætning",
            "denne her matche ikke den ovenstående",
        ]
        # "this is a test sentence", "this does not match the above"
        labels = [1, 0]
        data = DatasetDict(
            {
                "test": Dataset.from_dict(
                    {
                        "sentence1": sentence1,
                        "sentence2": sentence2,
                        "labels": labels,
                    }
                )
            }
        )

        self.dataset = {
            "eng": data,
            "fra": data,
        }
        self.data_loaded = True


class MockPairImageClassificationTask(AbsTaskPairClassification):
    expected_stats = {
        "test": {
            "num_samples": 2,
            "unique_pairs": 2,
            "number_of_characters": None,
            "text1_statistics": None,
            "image1_statistics": {
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
            "text2_statistics": None,
            "image2_statistics": {
                "min_image_width": 100,
                "average_image_width": 100.0,
                "max_image_width": 100,
                "min_image_height": 100,
                "average_image_height": 100.0,
                "max_image_height": 100,
                "unique_images": 2,
            },
            "audio2_statistics": None,
            "video2_statistics": None,
            "labels_statistics": {
                "min_labels_per_text": 1,
                "average_label_per_text": 1.0,
                "max_labels_per_text": 1,
                "unique_labels": 2,
                "labels": {"1": {"count": 1}, "0": {"count": 1}},
            },
        }
    }

    metadata = TaskMetadata(
        type="PairClassification",
        name="MockPairImageClassificationTask",
        main_score="similarity_ap",
        **general_args,
    )
    metadata.modalities = ["image"]
    metadata.category = "i2i"

    input1_column_name = "image1"
    input2_column_name = "image2"

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        images1 = create_mock_images(self.np_rng)
        images2 = create_mock_images(self.np_rng)

        labels = [1, 0]

        self.dataset = DatasetDict(
            {
                "test": Dataset.from_dict(
                    {
                        "image1": images1,
                        "image2": images2,
                        "labels": labels,
                    }
                ),
            }
        )
        self.data_loaded = True


class MockImageTextPairClassificationTask(AbsTaskImageTextPairClassification):
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
            "image_statistics": {
                "min_image_width": 100,
                "average_image_width": 100.0,
                "max_image_width": 100,
                "min_image_height": 100,
                "average_image_height": 100.0,
                "max_image_height": 100,
                "unique_images": 2,
            },
        }
    }

    metadata = TaskMetadata(
        type="Compositionality",
        name="MockImageTextPairClassification",
        main_score="text_acc",
        **general_args,
    )
    metadata.modalities = ["image", "text"]
    metadata.category = "i2t"

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        images = create_mock_images(self.np_rng)
        texts = ["This is a test sentence", "This is another test sentence"]

        self.dataset = DatasetDict(
            {
                "test": Dataset.from_dict(
                    {
                        "image": images,
                        "caption": texts,
                    }
                ),
            }
        )
        self.data_loaded = True


class MockMultilingualImageTextPairClassificationTask(
    AbsTaskImageTextPairClassification
):
    expected_stats = {
        "test": {
            "num_samples": 4,
            "text_statistics": {
                "total_text_length": 104,
                "min_text_length": 23,
                "average_text_length": 26.0,
                "max_text_length": 29,
                "unique_texts": 2,
            },
            "image_statistics": {
                "min_image_width": 100,
                "average_image_width": 100.0,
                "max_image_width": 100,
                "min_image_height": 100,
                "average_image_height": 100.0,
                "max_image_height": 100,
                "unique_images": 2,
            },
            "hf_subset_descriptive_stats": {
                "eng": {
                    "num_samples": 2,
                    "text_statistics": {
                        "total_text_length": 52,
                        "min_text_length": 23,
                        "average_text_length": 26.0,
                        "max_text_length": 29,
                        "unique_texts": 2,
                    },
                    "image_statistics": {
                        "min_image_width": 100,
                        "average_image_width": 100.0,
                        "max_image_width": 100,
                        "min_image_height": 100,
                        "average_image_height": 100.0,
                        "max_image_height": 100,
                        "unique_images": 2,
                    },
                },
                "fra": {
                    "num_samples": 2,
                    "text_statistics": {
                        "total_text_length": 52,
                        "min_text_length": 23,
                        "average_text_length": 26.0,
                        "max_text_length": 29,
                        "unique_texts": 2,
                    },
                    "image_statistics": {
                        "min_image_width": 100,
                        "average_image_width": 100.0,
                        "max_image_width": 100,
                        "min_image_height": 100,
                        "average_image_height": 100.0,
                        "max_image_height": 100,
                        "unique_images": 2,
                    },
                },
            },
        }
    }

    metadata = TaskMetadata(
        type="Compositionality",
        name="MockMultilingualImageTextPairClassification",
        main_score="accuracy",
        **general_args,
    )
    metadata.modalities = ["image", "text"]
    metadata.category = "i2t"

    metadata.eval_langs = multilingual_eval_langs

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        images = create_mock_images(self.np_rng)
        texts = ["This is a test sentence", "This is another test sentence"]
        data = {
            "test": Dataset.from_dict(
                {
                    "image": images,
                    "caption": texts,
                }
            )
        }

        self.dataset = DatasetDict(
            {
                "eng": data,
                "fra": data,
            }
        )
        self.data_loaded = True


class MockAudioPairClassification(AbsTaskPairClassification):
    metadata = TaskMetadata(
        type="AudioPairClassification",
        name="AbsTaskAudioPairClassification",
        main_score="max_ap",
        **general_args,  # type: ignore[arg-type]
    )
    metadata.modalities = ["audio"]
    metadata.category = "a2a"

    expected_stats = {
        "test": {
            "num_samples": 2,
            "unique_pairs": 2,
            "number_of_characters": None,
            "text1_statistics": None,
            "image1_statistics": None,
            "audio1_statistics": {
                "total_duration_seconds": 2.0,
                "min_duration_seconds": 1.0,
                "average_duration_seconds": 1.0,
                "max_duration_seconds": 1.0,
                "unique_audios": 2,
                "average_sampling_rate": 16000.0,
                "sampling_rates": {16000: 2},
            },
            "video1_statistics": None,
            "text2_statistics": None,
            "image2_statistics": None,
            "audio2_statistics": {
                "total_duration_seconds": 2.0,
                "min_duration_seconds": 1.0,
                "average_duration_seconds": 1.0,
                "max_duration_seconds": 1.0,
                "unique_audios": 2,
                "average_sampling_rate": 16000.0,
                "sampling_rates": {16000: 2},
            },
            "video2_statistics": None,
            "labels_statistics": {
                "min_labels_per_text": 1,
                "average_label_per_text": 1.0,
                "max_labels_per_text": 1,
                "unique_labels": 2,
                "labels": {"0": {"count": 1}, "1": {"count": 1}},
            },
        }
    }

    input1_column_name = "audio1"
    input2_column_name = "audio1"
    label_column_name = "label"

    def load_data(self, **kwargs):
        mock_audio = create_mock_audio(self.np_rng)

        self.dataset = DatasetDict(
            {
                "test": Dataset.from_dict(
                    {
                        "audio1": mock_audio,
                        "audio2": mock_audio,
                        "label": [0, 1],
                    }
                ),
            }
        )
        self.dataset = self.dataset.cast_column("audio1", Audio())
        self.dataset = self.dataset.cast_column("audio2", Audio())
        self.data_loaded = True


class MockVideoPairClassificationTask(AbsTaskPairClassification):
    metadata = TaskMetadata(
        type="VideoPairClassification",
        name="MockVideoPairClassification",
        main_score="max_ap",
        **general_args,  # type: ignore[arg-type]
    )
    metadata.modalities = ["video"]
    metadata.category = "v2v"

    expected_stats = {
        "test": {
            "num_samples": 2,
            "unique_pairs": 2,
            "number_of_characters": None,
            "text1_statistics": None,
            "image1_statistics": None,
            "audio1_statistics": None,
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
            "text2_statistics": None,
            "image2_statistics": None,
            "audio2_statistics": None,
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
            "labels_statistics": {
                "min_labels_per_text": 1,
                "average_label_per_text": 1.0,
                "max_labels_per_text": 1,
                "unique_labels": 2,
                "labels": {"0": {"count": 1}, "1": {"count": 1}},
            },
        }
    }

    input1_column_name = "video1"
    input2_column_name = "video2"
    label_column_name = "label"

    def load_data(self, **kwargs):
        from datasets import Video

        mock_videos = create_mock_video_bytes(self.np_rng)

        self.dataset = DatasetDict(
            {
                "test": Dataset.from_dict(
                    {
                        "video1": mock_videos,
                        "video2": mock_videos,
                        "label": [0, 1],
                    }
                ),
            }
        )
        self.dataset = self.dataset.cast_column("video1", Video())
        self.dataset = self.dataset.cast_column("video2", Video())
        self.data_loaded = True


class MockVideoAudioPairClassificationTask(AbsTaskPairClassification):
    metadata = TaskMetadata(
        type="VideoPairClassification",
        name="MockVideoAudioPairClassification",
        main_score="max_ap",
        **general_args,  # type: ignore[arg-type]
    )
    metadata.modalities = ["video", "audio"]
    metadata.category = "va2va"
    input1_column_name = {"video1": "video", "audio1": "audio"}
    input2_column_name = {"video2": "video", "audio2": "audio"}

    label_column_name = "label"

    expected_stats = {
        "test": {
            "num_samples": 2,
            "unique_pairs": 2,
            "number_of_characters": None,
            "text1_statistics": None,
            "image1_statistics": None,
            "audio1_statistics": {
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
            "text2_statistics": None,
            "image2_statistics": None,
            "audio2_statistics": {
                "total_duration_seconds": 2.0,
                "min_duration_seconds": 1.0,
                "average_duration_seconds": 1.0,
                "max_duration_seconds": 1.0,
                "unique_audios": 2,
                "average_sampling_rate": 16000.0,
                "sampling_rates": {16000: 2},
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
            "labels_statistics": {
                "min_labels_per_text": 1,
                "average_label_per_text": 1.0,
                "max_labels_per_text": 1,
                "unique_labels": 2,
                "labels": {"0": {"count": 1}, "1": {"count": 1}},
            },
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
                        "video1": mock_videos,
                        "audio1": mock_audio,
                        "video2": mock_videos,
                        "audio2": mock_audio,
                        "label": [0, 1],
                    }
                ),
            }
        )
        self.dataset = self.dataset.cast_column("video1", Video())
        self.dataset = self.dataset.cast_column("video2", Video())
        self.dataset = self.dataset.cast_column("audio1", Audio())
        self.dataset = self.dataset.cast_column("audio2", Audio())
        self.data_loaded = True


class MockAsymVideoAudioPairClassificationTask(AbsTaskPairClassification):
    """Asymmetric pair classification: side-1 is video only, side-2 is audio only."""

    metadata = TaskMetadata(
        type="VideoPairClassification",
        name="MockAsymVideoAudioPairClassificationTask",
        main_score="max_ap",
        **general_args,  # type: ignore[arg-type]
    )
    metadata.modalities = ["video", "audio"]
    metadata.category = "v2a"

    input1_column_name = {"video1": "video"}
    input2_column_name = {"audio2": "audio"}
    input1_prompt_type = PromptType.query
    input2_prompt_type = PromptType.document

    label_column_name = "label"

    expected_stats = {
        "test": {
            "num_samples": 2,
            "unique_pairs": 2,
            "number_of_characters": None,
            "text1_statistics": None,
            "image1_statistics": None,
            "audio1_statistics": None,
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
            "text2_statistics": None,
            "image2_statistics": None,
            "audio2_statistics": {
                "total_duration_seconds": 2.0,
                "min_duration_seconds": 1.0,
                "average_duration_seconds": 1.0,
                "max_duration_seconds": 1.0,
                "unique_audios": 2,
                "average_sampling_rate": 16000.0,
                "sampling_rates": {16000: 2},
            },
            "video2_statistics": None,
            "labels_statistics": {
                "min_labels_per_text": 1,
                "average_label_per_text": 1.0,
                "max_labels_per_text": 1,
                "unique_labels": 2,
                "labels": {"0": {"count": 1}, "1": {"count": 1}},
            },
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
                        "video1": mock_videos,
                        "audio2": mock_audio,
                        "label": [0, 1],
                    }
                ),
            }
        )
        self.dataset = self.dataset.cast_column("video1", Video())
        self.dataset = self.dataset.cast_column("audio2", Audio())
        self.data_loaded = True


class MockAsymVideoAudioPairClassificationTaskV2(AbsTaskPairClassification):
    """Asymmetric pair classification: side-1 is video only, side-2 is audio only. Differs from v1 by `input_column_name` is str instead of list"""

    metadata = TaskMetadata(
        type="VideoPairClassification",
        name="MockAsymVideoAudioPairClassificationTaskV2",
        main_score="max_ap",
        **general_args,  # type: ignore[arg-type]
    )
    metadata.modalities = ["video", "audio"]
    metadata.category = "v2a"

    input1_column_name = {"video": "video"}
    input2_column_name = {"audio": "audio"}
    input1_prompt_type = PromptType.query
    input2_prompt_type = PromptType.document

    label_column_name = "label"

    expected_stats = {
        "test": {
            "num_samples": 2,
            "unique_pairs": 2,
            "number_of_characters": None,
            "text1_statistics": None,
            "image1_statistics": None,
            "audio1_statistics": None,
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
            "text2_statistics": None,
            "image2_statistics": None,
            "audio2_statistics": {
                "total_duration_seconds": 2.0,
                "min_duration_seconds": 1.0,
                "average_duration_seconds": 1.0,
                "max_duration_seconds": 1.0,
                "unique_audios": 2,
                "average_sampling_rate": 16000.0,
                "sampling_rates": {16000: 2},
            },
            "video2_statistics": None,
            "labels_statistics": {
                "min_labels_per_text": 1,
                "average_label_per_text": 1.0,
                "max_labels_per_text": 1,
                "unique_labels": 2,
                "labels": {"0": {"count": 1}, "1": {"count": 1}},
            },
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
                        "label": [0, 1],
                    }
                ),
            }
        )
        self.dataset = self.dataset.cast_column("video", Video())
        self.dataset = self.dataset.cast_column("audio", Audio())
        self.data_loaded = True


class MockSymCustomVideoAudioPairClassificationTaskV2(AbsTaskPairClassification):
    """Asymmetric pair classification: side-1 is video only, side-2 is audio only. Differs from v1 by `input_column_name` is str instead of list"""

    metadata = TaskMetadata(
        type="VideoPairClassification",
        name="MockSymCustomVideoAudioPairClassificationTaskV2",
        main_score="max_ap",
        **general_args,  # type: ignore[arg-type]
    )
    metadata.modalities = ["video", "audio"]
    metadata.category = "v2a"

    input1_column_name = {"video": "video"}
    input2_column_name = {"audio": "audio"}
    input1_prompt_type = PromptType.document
    input2_prompt_type = PromptType.document

    label_column_name = "label"

    expected_stats = {
        "test": {
            "num_samples": 2,
            "unique_pairs": 2,
            "number_of_characters": None,
            "text1_statistics": None,
            "image1_statistics": None,
            "audio1_statistics": None,
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
            "text2_statistics": None,
            "image2_statistics": None,
            "audio2_statistics": {
                "total_duration_seconds": 2.0,
                "min_duration_seconds": 1.0,
                "average_duration_seconds": 1.0,
                "max_duration_seconds": 1.0,
                "unique_audios": 2,
                "average_sampling_rate": 16000.0,
                "sampling_rates": {16000: 2},
            },
            "video2_statistics": None,
            "labels_statistics": {
                "min_labels_per_text": 1,
                "average_label_per_text": 1.0,
                "max_labels_per_text": 1,
                "unique_labels": 2,
                "labels": {"0": {"count": 1}, "1": {"count": 1}},
            },
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
                        "label": [0, 1],
                    }
                ),
            }
        )
        self.dataset = self.dataset.cast_column("video", Video())
        self.dataset = self.dataset.cast_column("audio", Audio())
        self.data_loaded = True
