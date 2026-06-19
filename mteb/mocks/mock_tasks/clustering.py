from __future__ import annotations

from datasets import Audio, Dataset, DatasetDict

from mteb.abstasks.clustering import AbsTaskClustering
from mteb.abstasks.clustering_legacy import AbsTaskClusteringLegacy
from mteb.abstasks.task_metadata import TaskMetadata

from .utils import (
    create_mock_audio,
    create_mock_images,
    create_mock_video_bytes,
    general_args,
    multilingual_eval_langs,
)


class MockClusteringTask(AbsTaskClusteringLegacy):
    expected_stats = {
        "test": {
            "num_samples": 3,
            "text_statistics": {
                "total_text_length": 81,
                "min_text_length": 23,
                "average_text_length": 27.0,
                "max_text_length": 29,
                "unique_texts": 3,
            },
            "image_statistics": None,
            "audio_statistics": None,
            "video_statistics": None,
            "label_statistics": {
                "min_labels_per_text": 1,
                "average_label_per_text": 1.0,
                "max_labels_per_text": 1,
                "unique_labels": 3,
                "labels": {"0": {"count": 1}, "1": {"count": 1}, "2": {"count": 1}},
            },
        }
    }

    metadata = TaskMetadata(
        type="Clustering",
        name="MockClusteringTask",
        main_score="v_measure",
        **general_args,
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        sentences = [
            [
                "This is a test sentence",
                "This is another test sentence",
                "This is a third test sentence",
            ]
        ]
        labels = [[0, 1, 2]]

        self.dataset = DatasetDict(
            {
                "test": Dataset.from_dict(
                    {
                        "sentences": sentences,
                        "labels": labels,
                    }
                ),
            }
        )
        self.data_loaded = True


class MockMultilingualClusteringTask(AbsTaskClusteringLegacy):
    expected_stats = {
        "test": {
            "num_samples": 6,
            "text_statistics": {
                "total_text_length": 162,
                "min_text_length": 23,
                "average_text_length": 27.0,
                "max_text_length": 29,
                "unique_texts": 3,
            },
            "image_statistics": None,
            "audio_statistics": None,
            "video_statistics": None,
            "label_statistics": {
                "min_labels_per_text": 1,
                "average_label_per_text": 1.0,
                "max_labels_per_text": 1,
                "unique_labels": 3,
                "labels": {"0": {"count": 2}, "1": {"count": 2}, "2": {"count": 2}},
            },
            "hf_subset_descriptive_stats": {
                "eng": {
                    "num_samples": 3,
                    "text_statistics": {
                        "total_text_length": 81,
                        "min_text_length": 23,
                        "average_text_length": 27.0,
                        "max_text_length": 29,
                        "unique_texts": 3,
                    },
                    "image_statistics": None,
                    "audio_statistics": None,
                    "video_statistics": None,
                    "label_statistics": {
                        "min_labels_per_text": 1,
                        "average_label_per_text": 1.0,
                        "max_labels_per_text": 1,
                        "unique_labels": 3,
                        "labels": {
                            "0": {"count": 1},
                            "1": {"count": 1},
                            "2": {"count": 1},
                        },
                    },
                },
                "fra": {
                    "num_samples": 3,
                    "text_statistics": {
                        "total_text_length": 81,
                        "min_text_length": 23,
                        "average_text_length": 27.0,
                        "max_text_length": 29,
                        "unique_texts": 3,
                    },
                    "image_statistics": None,
                    "audio_statistics": None,
                    "video_statistics": None,
                    "label_statistics": {
                        "min_labels_per_text": 1,
                        "average_label_per_text": 1.0,
                        "max_labels_per_text": 1,
                        "unique_labels": 3,
                        "labels": {
                            "0": {"count": 1},
                            "1": {"count": 1},
                            "2": {"count": 1},
                        },
                    },
                },
            },
        }
    }

    metadata = TaskMetadata(
        type="Clustering",
        name="MockMultilingualClusteringTask",
        main_score="v_measure",
        **general_args,
    )
    metadata.eval_langs = multilingual_eval_langs

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        sentences = [
            [
                "This is a test sentence",
                "This is another test sentence",
                "This is a third test sentence",
            ]
        ]
        labels = [[0, 1, 2]]
        data = {
            "test": Dataset.from_dict(
                {
                    "sentences": sentences,
                    "labels": labels,
                }
            ),
        }

        self.dataset = DatasetDict(
            {
                "eng": data,
                "fra": data,
            }
        )
        self.data_loaded = True


class MockClusteringFastTask(AbsTaskClustering):
    max_document_to_embed = 20
    max_fraction_of_documents_to_embed = None
    n_clusters = 3
    max_documents_per_cluster = 4
    expected_stats = {
        "test": {
            "num_samples": 20,
            "text_statistics": {
                "total_text_length": 550,
                "min_text_length": 23,
                "average_text_length": 27.5,
                "max_text_length": 29,
                "unique_texts": 3,
            },
            "image_statistics": None,
            "audio_statistics": None,
            "video_statistics": None,
            "labels_statistics": {
                "min_labels_per_text": 1,
                "average_label_per_text": 1.0,
                "max_labels_per_text": 1,
                "unique_labels": 3,
                "labels": {"0": {"count": 5}, "1": {"count": 5}, "2": {"count": 10}},
            },
        }
    }

    metadata = TaskMetadata(
        type="Clustering",
        name="MockClusteringFastTask",
        main_score="v_measure",
        **general_args,
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        sentences = [
            "This is a test sentence",
            "This is another test sentence",
            "This is a third test sentence",
            "This is a third test sentence",
        ] * 5
        labels = [0, 1, 2, 2] * 5

        self.dataset = DatasetDict(
            {
                "test": Dataset.from_dict(
                    {
                        "sentences": sentences,
                        "labels": labels,
                    }
                ),
            }
        )
        self.data_loaded = True


class MockMultilingualClusteringFastTask(AbsTaskClustering):
    max_document_to_embed = 3
    max_fraction_of_documents_to_embed = None
    expected_stats = {
        "test": {
            "num_samples": 6,
            "text_statistics": {
                "total_text_length": 162,
                "min_text_length": 23,
                "average_text_length": 27.0,
                "max_text_length": 29,
                "unique_texts": 3,
            },
            "image_statistics": None,
            "audio_statistics": None,
            "video_statistics": None,
            "labels_statistics": {
                "min_labels_per_text": 1,
                "average_label_per_text": 1.0,
                "max_labels_per_text": 1,
                "unique_labels": 3,
                "labels": {"0": {"count": 2}, "1": {"count": 2}, "2": {"count": 2}},
            },
            "hf_subset_descriptive_stats": {
                "eng": {
                    "num_samples": 3,
                    "text_statistics": {
                        "total_text_length": 81,
                        "min_text_length": 23,
                        "average_text_length": 27.0,
                        "max_text_length": 29,
                        "unique_texts": 3,
                    },
                    "image_statistics": None,
                    "audio_statistics": None,
                    "video_statistics": None,
                    "labels_statistics": {
                        "min_labels_per_text": 1,
                        "average_label_per_text": 1.0,
                        "max_labels_per_text": 1,
                        "unique_labels": 3,
                        "labels": {
                            "0": {"count": 1},
                            "1": {"count": 1},
                            "2": {"count": 1},
                        },
                    },
                },
                "fra": {
                    "num_samples": 3,
                    "text_statistics": {
                        "total_text_length": 81,
                        "min_text_length": 23,
                        "average_text_length": 27.0,
                        "max_text_length": 29,
                        "unique_texts": 3,
                    },
                    "image_statistics": None,
                    "audio_statistics": None,
                    "video_statistics": None,
                    "labels_statistics": {
                        "min_labels_per_text": 1,
                        "average_label_per_text": 1.0,
                        "max_labels_per_text": 1,
                        "unique_labels": 3,
                        "labels": {
                            "0": {"count": 1},
                            "1": {"count": 1},
                            "2": {"count": 1},
                        },
                    },
                },
            },
        }
    }

    metadata = TaskMetadata(
        type="Clustering",
        name="MockMultilingualClusteringFastTask",
        main_score="v_measure",
        **general_args,
    )
    metadata.eval_langs = multilingual_eval_langs

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        sentences = [
            "This is a test sentence",
            "This is another test sentence",
            "This is a third test sentence",
        ]
        labels = [0, 1, 2]
        data = {
            "test": Dataset.from_dict(
                {
                    "sentences": sentences,
                    "labels": labels,
                }
            ),
        }

        self.dataset = DatasetDict(
            {
                "eng": data,
                "fra": data,
            }
        )
        self.data_loaded = True


class MockImageClusteringTask(AbsTaskClusteringLegacy):
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
                "labels": {"1": {"count": 1}, "0": {"count": 1}},
            },
        }
    }

    metadata = TaskMetadata(
        type="ImageClustering",
        name="MockImageClustering",
        main_score="nmi",
        **general_args,
    )
    metadata.modalities = ["image"]
    input_column_name = "image"
    label_column_name = "label"

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        images = create_mock_images(self.np_rng)
        labels = [1, 0]

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


class MockImageClusteringFastTask(AbsTaskClustering):
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
        type="ImageClustering",
        name="MockImageClusteringFastTask",
        main_score="v_measure",
        **general_args,
    )
    metadata.modalities = ["image"]
    input_column_name = "image"
    label_column_name = "label"
    max_fraction_of_documents_to_embed = None
    max_document_to_embed = 2

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        images = create_mock_images(self.np_rng)
        labels = [1, 0]

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


class MockAudioClusteringTask(AbsTaskClustering):
    max_document_to_embed = 2
    max_fraction_of_documents_to_embed = None
    input_column_name = "audio"

    expected_stats = {
        "test": {
            "num_samples": 3,
            "text_statistics": None,
            "image_statistics": None,
            "audio_statistics": {
                "total_duration_seconds": 3.0,
                "min_duration_seconds": 1.0,
                "average_duration_seconds": 1.0,
                "max_duration_seconds": 1.0,
                "unique_audios": 3,
                "average_sampling_rate": 16000.0,
                "sampling_rates": {16000: 3},
            },
            "video_statistics": None,
            "labels_statistics": {
                "min_labels_per_text": 1,
                "average_label_per_text": 1.0,
                "max_labels_per_text": 1,
                "unique_labels": 3,
                "labels": {"0": {"count": 1}, "1": {"count": 1}, "2": {"count": 1}},
            },
        }
    }

    metadata = TaskMetadata(
        type="Clustering",
        name="MockAudioClusteringTask",
        main_score="v_measure",
        **general_args,
    )
    metadata.modalities = ["audio"]

    def load_data(self, **kwargs):
        mock_audio = create_mock_audio(self.np_rng, n=3)

        labels = [0, 1, 2]

        self.dataset = DatasetDict(
            {
                "test": Dataset.from_dict(
                    {
                        "audio": mock_audio,
                        "labels": labels,
                    }
                ),
            }
        )
        self.dataset = self.dataset.cast_column("audio", Audio())
        self.data_loaded = True


class MockVideoClusteringTask(AbsTaskClustering):
    max_document_to_embed = 3
    max_fraction_of_documents_to_embed = None
    input_column_name = "video"

    expected_stats = {
        "test": {
            "num_samples": 3,
            "text_statistics": None,
            "image_statistics": None,
            "audio_statistics": None,
            "video_statistics": {
                "total_duration_seconds": 3.0,
                "total_frames": 72,
                "min_width": 64,
                "average_width": 64.0,
                "max_width": 64,
                "min_height": 64,
                "average_height": 64.0,
                "max_height": 64,
                "min_duration_seconds": 1.0,
                "average_duration_seconds": 1.0,
                "max_duration_seconds": 1.0,
                "unique_videos": 3,
                "average_fps": 24.0,
                "fps": {24: 3},
                "min_resolution": (64, 64),
                "average_resolution": (64.0, 64.0),
                "max_resolution": (64, 64),
                "resolutions": {"64x64": 3},
            },
            "labels_statistics": {
                "min_labels_per_text": 1,
                "average_label_per_text": 1.0,
                "max_labels_per_text": 1,
                "unique_labels": 3,
                "labels": {"0": {"count": 1}, "1": {"count": 1}, "2": {"count": 1}},
            },
        }
    }

    metadata = TaskMetadata(
        type="VideoClustering",
        name="MockVideoClusteringTask",
        main_score="v_measure",
        **general_args,
    )
    metadata.modalities = ["video"]
    metadata.category = "v2c"

    def load_data(self, **kwargs):
        from datasets import Video

        mock_videos = create_mock_video_bytes(self.np_rng, n=3)

        self.dataset = DatasetDict(
            {
                "test": Dataset.from_dict(
                    {
                        "video": mock_videos,
                        "labels": [0, 1, 2],
                    }
                ),
            }
        )
        self.dataset = self.dataset.cast_column("video", Video())
        self.data_loaded = True


class MockVideoAudioClusteringTask(AbsTaskClustering):
    max_document_to_embed = 3
    max_fraction_of_documents_to_embed = None
    input_column_name = ("video", "audio")
    label_column_name = "labels"

    expected_stats = {
        "test": {
            "num_samples": 3,
            "text_statistics": None,
            "image_statistics": None,
            "audio_statistics": {
                "total_duration_seconds": 3.0,
                "min_duration_seconds": 1.0,
                "average_duration_seconds": 1.0,
                "max_duration_seconds": 1.0,
                "unique_audios": 3,
                "average_sampling_rate": 16000.0,
                "sampling_rates": {16000: 3},
            },
            "video_statistics": {
                "total_duration_seconds": 3.0,
                "total_frames": 72,
                "min_width": 64,
                "average_width": 64.0,
                "max_width": 64,
                "min_height": 64,
                "average_height": 64.0,
                "max_height": 64,
                "min_duration_seconds": 1.0,
                "average_duration_seconds": 1.0,
                "max_duration_seconds": 1.0,
                "unique_videos": 3,
                "average_fps": 24.0,
                "fps": {24: 3},
                "min_resolution": (64, 64),
                "average_resolution": (64.0, 64.0),
                "max_resolution": (64, 64),
                "resolutions": {"64x64": 3},
            },
            "labels_statistics": {
                "min_labels_per_text": 1,
                "average_label_per_text": 1.0,
                "max_labels_per_text": 1,
                "unique_labels": 3,
                "labels": {"0": {"count": 1}, "1": {"count": 1}, "2": {"count": 1}},
            },
        }
    }

    metadata = TaskMetadata(
        type="VideoClustering",
        name="MockVideoAudioClusteringTask",
        main_score="v_measure",
        **general_args,
    )
    metadata.modalities = ["video", "audio"]
    metadata.category = "va2c"

    def load_data(self, **kwargs):
        from datasets import Video

        mock_videos = create_mock_video_bytes(self.np_rng, n=3)
        mock_audio = create_mock_audio(self.np_rng, n=3)

        self.dataset = DatasetDict(
            {
                "test": Dataset.from_dict(
                    {
                        "video": mock_videos,
                        "audio": mock_audio,
                        "labels": [0, 1, 2],
                    }
                ),
            }
        )

        self.dataset = self.dataset.cast_column("video", Video())
        self.dataset = self.dataset.cast_column("audio", Audio())
        self.data_loaded = True
