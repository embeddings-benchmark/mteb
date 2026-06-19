from __future__ import annotations

from datasets import Audio, Dataset, DatasetDict
from sklearn.linear_model import LogisticRegression

from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.multilabel_classification import (
    AbsTaskMultilabelClassification,
)
from mteb.abstasks.task_metadata import TaskMetadata

from .utils import (
    create_mock_audio,
    create_mock_images,
    create_mock_video_bytes,
    general_args,
    multilingual_eval_langs,
)


class MockClassificationTask(AbsTaskClassification):
    classifier = LogisticRegression(max_iter=10)

    expected_stats = {
        "test": {
            "num_samples": 2,
            "samples_in_train": 1,
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
                "labels": {"0": {"count": 1}, "1": {"count": 1}},
            },
        },
        "train": {
            "num_samples": 2,
            "samples_in_train": None,
            "text_statistics": {
                "total_text_length": 53,
                "min_text_length": 23,
                "average_text_length": 26.5,
                "max_text_length": 30,
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
                "labels": {"0": {"count": 1}, "1": {"count": 1}},
            },
        },
    }

    metadata = TaskMetadata(
        type="Classification",
        name="MockClassificationTask",
        main_score="accuracy",
        **general_args,
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        train_texts = ["This is a test sentence", "This is another train sentence"]
        test_texts = ["This is a test sentence", "This is another test sentence"]

        labels = [0, 1]

        self.dataset = DatasetDict(
            {
                "test": Dataset.from_dict(
                    {
                        "text": test_texts,
                        "label": labels,
                    }
                ),
                "train": Dataset.from_dict(
                    {
                        "text": train_texts,
                        "label": labels,
                    }
                ),
            }
        )
        self.data_loaded = True


class MockMultilingualClassificationTask(AbsTaskClassification):
    classifier = LogisticRegression(max_iter=10)

    expected_stats = {
        "test": {
            "num_samples": 4,
            "samples_in_train": 1,
            "text_statistics": {
                "total_text_length": 104,
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
                "labels": {"0": {"count": 2}, "1": {"count": 2}},
            },
            "hf_subset_descriptive_stats": {
                "eng": {
                    "num_samples": 2,
                    "samples_in_train": 1,
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
                        "labels": {"0": {"count": 1}, "1": {"count": 1}},
                    },
                },
                "fra": {
                    "num_samples": 2,
                    "samples_in_train": 1,
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
                        "labels": {"0": {"count": 1}, "1": {"count": 1}},
                    },
                },
            },
        },
        "train": {
            "num_samples": 4,
            "samples_in_train": None,
            "text_statistics": {
                "total_text_length": 106,
                "min_text_length": 23,
                "average_text_length": 26.5,
                "max_text_length": 30,
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
                "labels": {"0": {"count": 2}, "1": {"count": 2}},
            },
            "hf_subset_descriptive_stats": {
                "eng": {
                    "num_samples": 2,
                    "samples_in_train": None,
                    "text_statistics": {
                        "total_text_length": 53,
                        "min_text_length": 23,
                        "average_text_length": 26.5,
                        "max_text_length": 30,
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
                        "labels": {"0": {"count": 1}, "1": {"count": 1}},
                    },
                },
                "fra": {
                    "num_samples": 2,
                    "samples_in_train": None,
                    "text_statistics": {
                        "total_text_length": 53,
                        "min_text_length": 23,
                        "average_text_length": 26.5,
                        "max_text_length": 30,
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
                        "labels": {"0": {"count": 1}, "1": {"count": 1}},
                    },
                },
            },
        },
    }

    metadata = TaskMetadata(
        type="Classification",
        name="MockMultilingualClassificationTask",
        main_score="accuracy",
        **general_args,
    )
    metadata.eval_langs = multilingual_eval_langs

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        train_texts = ["This is a test sentence", "This is another train sentence"]
        test_texts = ["This is a test sentence", "This is another test sentence"]
        labels = [0, 1]
        data = {
            "test": Dataset.from_dict(
                {
                    "text": test_texts,
                    "label": labels,
                }
            ),
            "train": Dataset.from_dict(
                {
                    "text": train_texts,
                    "label": labels,
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


class MockMultilabelClassification(AbsTaskMultilabelClassification):
    expected_stats = {
        "test": {
            "num_samples": 6,
            "samples_in_train": 1,
            "text_statistics": {
                "total_text_length": 156,
                "min_text_length": 23,
                "average_text_length": 26.0,
                "max_text_length": 29,
                "unique_texts": 2,
            },
            "image_statistics": None,
            "audio_statistics": None,
            "video_statistics": None,
            "label_statistics": {
                "min_labels_per_text": 2,
                "average_label_per_text": 2.0,
                "max_labels_per_text": 2,
                "unique_labels": 2,
                "labels": {"0": {"count": 6}, "1": {"count": 6}},
            },
        },
        "train": {
            "num_samples": 6,
            "samples_in_train": None,
            "text_statistics": {
                "total_text_length": 159,
                "min_text_length": 23,
                "average_text_length": 26.5,
                "max_text_length": 30,
                "unique_texts": 2,
            },
            "image_statistics": None,
            "audio_statistics": None,
            "video_statistics": None,
            "label_statistics": {
                "min_labels_per_text": 2,
                "average_label_per_text": 2.0,
                "max_labels_per_text": 2,
                "unique_labels": 2,
                "labels": {"0": {"count": 6}, "1": {"count": 6}},
            },
        },
    }

    metadata = TaskMetadata(
        type="MultilabelClassification",
        name="MockMultilabelClassification",
        main_score="lrap",
        **general_args,
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        train_texts = ["This is a test sentence", "This is another train sentence"] * 3
        test_texts = ["This is a test sentence", "This is another test sentence"] * 3
        labels = [[0, 1], [1, 0]] * 3

        self.dataset = DatasetDict(
            {
                "test": Dataset.from_dict(
                    {
                        "text": test_texts,
                        "label": labels,
                    }
                ),
                "train": Dataset.from_dict(
                    {
                        "text": train_texts,
                        "label": labels,
                    }
                ),
            }
        )
        self.data_loaded = True


class MockMultilingualMultilabelClassification(AbsTaskMultilabelClassification):
    expected_stats = {
        "test": {
            "num_samples": 12,
            "samples_in_train": 1,
            "text_statistics": {
                "total_text_length": 312,
                "min_text_length": 23,
                "average_text_length": 26.0,
                "max_text_length": 29,
                "unique_texts": 2,
            },
            "image_statistics": None,
            "audio_statistics": None,
            "video_statistics": None,
            "label_statistics": {
                "min_labels_per_text": 2,
                "average_label_per_text": 2.0,
                "max_labels_per_text": 2,
                "unique_labels": 2,
                "labels": {"0": {"count": 12}, "1": {"count": 12}},
            },
            "hf_subset_descriptive_stats": {
                "eng": {
                    "num_samples": 6,
                    "samples_in_train": 1,
                    "text_statistics": {
                        "total_text_length": 156,
                        "min_text_length": 23,
                        "average_text_length": 26.0,
                        "max_text_length": 29,
                        "unique_texts": 2,
                    },
                    "image_statistics": None,
                    "audio_statistics": None,
                    "video_statistics": None,
                    "label_statistics": {
                        "min_labels_per_text": 2,
                        "average_label_per_text": 2.0,
                        "max_labels_per_text": 2,
                        "unique_labels": 2,
                        "labels": {"0": {"count": 6}, "1": {"count": 6}},
                    },
                },
                "fra": {
                    "num_samples": 6,
                    "samples_in_train": 1,
                    "text_statistics": {
                        "total_text_length": 156,
                        "min_text_length": 23,
                        "average_text_length": 26.0,
                        "max_text_length": 29,
                        "unique_texts": 2,
                    },
                    "image_statistics": None,
                    "audio_statistics": None,
                    "video_statistics": None,
                    "label_statistics": {
                        "min_labels_per_text": 2,
                        "average_label_per_text": 2.0,
                        "max_labels_per_text": 2,
                        "unique_labels": 2,
                        "labels": {"0": {"count": 6}, "1": {"count": 6}},
                    },
                },
            },
        },
        "train": {
            "num_samples": 12,
            "samples_in_train": None,
            "text_statistics": {
                "total_text_length": 318,
                "min_text_length": 23,
                "average_text_length": 26.5,
                "max_text_length": 30,
                "unique_texts": 2,
            },
            "image_statistics": None,
            "audio_statistics": None,
            "video_statistics": None,
            "label_statistics": {
                "min_labels_per_text": 2,
                "average_label_per_text": 2.0,
                "max_labels_per_text": 2,
                "unique_labels": 2,
                "labels": {"0": {"count": 12}, "1": {"count": 12}},
            },
            "hf_subset_descriptive_stats": {
                "eng": {
                    "num_samples": 6,
                    "samples_in_train": None,
                    "text_statistics": {
                        "total_text_length": 159,
                        "min_text_length": 23,
                        "average_text_length": 26.5,
                        "max_text_length": 30,
                        "unique_texts": 2,
                    },
                    "image_statistics": None,
                    "audio_statistics": None,
                    "video_statistics": None,
                    "label_statistics": {
                        "min_labels_per_text": 2,
                        "average_label_per_text": 2.0,
                        "max_labels_per_text": 2,
                        "unique_labels": 2,
                        "labels": {"0": {"count": 6}, "1": {"count": 6}},
                    },
                },
                "fra": {
                    "num_samples": 6,
                    "samples_in_train": None,
                    "text_statistics": {
                        "total_text_length": 159,
                        "min_text_length": 23,
                        "average_text_length": 26.5,
                        "max_text_length": 30,
                        "unique_texts": 2,
                    },
                    "image_statistics": None,
                    "audio_statistics": None,
                    "video_statistics": None,
                    "label_statistics": {
                        "min_labels_per_text": 2,
                        "average_label_per_text": 2.0,
                        "max_labels_per_text": 2,
                        "unique_labels": 2,
                        "labels": {"0": {"count": 6}, "1": {"count": 6}},
                    },
                },
            },
        },
    }

    metadata = TaskMetadata(
        type="MultilabelClassification",
        name="MockMultilingualMultilabelClassification",
        main_score="lrap",
        **general_args,
    )
    metadata.eval_langs = multilingual_eval_langs

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        train_texts = ["This is a test sentence", "This is another train sentence"] * 3
        test_texts = ["This is a test sentence", "This is another test sentence"] * 3
        labels = [[0, 1], [1, 0]] * 3

        data = {
            "test": Dataset.from_dict(
                {
                    "text": test_texts,
                    "label": labels,
                }
            ),
            "train": Dataset.from_dict(
                {
                    "text": train_texts,
                    "label": labels,
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


class MockImageClassificationTask(AbsTaskClassification):
    expected_stats = {
        "test": {
            "num_samples": 2,
            "samples_in_train": 2,
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
        },
        "train": {
            "num_samples": 10,
            "samples_in_train": None,
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
                "labels": {"1": {"count": 5}, "0": {"count": 5}},
            },
        },
    }

    metadata = TaskMetadata(
        type="ImageClassification",
        name="MockImageClassification",
        main_score="accuracy",
        **general_args,
    )
    metadata.modalities = ["image"]
    metadata.category = "i2c"
    n_experiments = 1
    samples_per_label = 5
    input_column_name = "image"

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
                "train": Dataset.from_dict(
                    {
                        "image": images * 5,
                        "label": labels * 5,
                    }
                ),
            }
        )
        self.data_loaded = True


class MockMultilingualImageClassificationTask(AbsTaskClassification):
    n_experiments = 1
    samples_per_label = 5
    expected_stats = {
        "test": {
            "num_samples": 4,
            "samples_in_train": 2,
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
                "labels": {"1": {"count": 2}, "0": {"count": 2}},
            },
            "hf_subset_descriptive_stats": {
                "eng": {
                    "num_samples": 2,
                    "samples_in_train": 2,
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
                },
                "fra": {
                    "num_samples": 2,
                    "samples_in_train": 2,
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
                },
            },
        },
        "train": {
            "num_samples": 20,
            "samples_in_train": None,
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
                "labels": {"1": {"count": 10}, "0": {"count": 10}},
            },
            "hf_subset_descriptive_stats": {
                "eng": {
                    "num_samples": 10,
                    "samples_in_train": None,
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
                        "labels": {"1": {"count": 5}, "0": {"count": 5}},
                    },
                },
                "fra": {
                    "num_samples": 10,
                    "samples_in_train": None,
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
                        "labels": {"1": {"count": 5}, "0": {"count": 5}},
                    },
                },
            },
        },
    }

    metadata = TaskMetadata(
        type="ImageClassification",
        name="MockMultilingualImageClassification",
        main_score="accuracy",
        **general_args,
    )
    metadata.modalities = ["image"]
    metadata.category = "i2c"
    metadata.eval_langs = multilingual_eval_langs
    input_column_name = "image"

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        images = create_mock_images(self.np_rng)
        labels = [1, 0]
        data = {
            "test": Dataset.from_dict(
                {
                    "image": images,
                    "label": labels,
                }
            ),
            "train": Dataset.from_dict(
                {
                    "image": images * 5,
                    "label": labels * 5,
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


class MockImageMultilabelClassificationTask(AbsTaskMultilabelClassification):
    expected_stats = {
        "test": {
            "num_samples": 4,
            "samples_in_train": 2,
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
                "min_labels_per_text": 2,
                "average_label_per_text": 2.0,
                "max_labels_per_text": 2,
                "unique_labels": 4,
                "labels": {
                    "0": {"count": 2},
                    "3": {"count": 2},
                    "1": {"count": 2},
                    "2": {"count": 2},
                },
            },
        },
        "train": {
            "num_samples": 10,
            "samples_in_train": None,
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
                "min_labels_per_text": 2,
                "average_label_per_text": 2.0,
                "max_labels_per_text": 2,
                "unique_labels": 4,
                "labels": {
                    "0": {"count": 5},
                    "3": {"count": 5},
                    "1": {"count": 5},
                    "2": {"count": 5},
                },
            },
        },
    }

    metadata = TaskMetadata(
        type="MultilabelClassification",
        name="MockImageMultilabelClassification",
        main_score="accuracy",
        **general_args,
    )
    metadata.modalities = ["image"]
    metadata.category = "i2c"
    n_experiments = 1
    samples_per_label = 3
    input_column_name = "image"

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        images = create_mock_images(self.np_rng)
        labels = [["0", "3"], ["1", "2"]]

        self.dataset = DatasetDict(
            {
                "test": Dataset.from_dict(
                    {
                        "image": images * 2,
                        "label": labels * 2,
                    }
                ),
                "train": Dataset.from_dict(
                    {
                        "image": images * 5,
                        "label": labels * 5,
                    }
                ),
            }
        )
        self.data_loaded = True


class MockAudioMultilabelClassificationTask(AbsTaskMultilabelClassification):
    expected_stats = {
        "test": {
            "num_samples": 2,
            "samples_in_train": 2,
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
        },
        "train": {
            "num_samples": 10,
            "samples_in_train": None,
            "text_statistics": None,
            "image_statistics": None,
            "audio_statistics": {
                "total_duration_seconds": 10.0,
                "min_duration_seconds": 1.0,
                "average_duration_seconds": 1.0,
                "max_duration_seconds": 1.0,
                "unique_audios": 2,
                "average_sampling_rate": 16000.0,
                "sampling_rates": {16000: 10},
            },
            "video_statistics": None,
            "label_statistics": {
                "min_labels_per_text": 1,
                "average_label_per_text": 1.0,
                "max_labels_per_text": 1,
                "unique_labels": 2,
                "labels": {"0": {"count": 5}, "1": {"count": 5}},
            },
        },
    }

    metadata = TaskMetadata(
        type="AudioMultilabelClassification",
        name="MockAudioMultilabelClassification",
        main_score="accuracy",
        **general_args,
    )
    metadata.modalities = ["audio"]
    input_column_name = "audio"

    def load_data(self, **kwargs):
        mock_audio = create_mock_audio(self.np_rng)
        labels = [[0], [1]]

        self.dataset = DatasetDict(
            {
                "test": Dataset.from_dict({"audio": mock_audio, "label": labels}),
                "train": Dataset.from_dict(
                    {"audio": mock_audio * 5, "label": labels * 5}
                ),
            }
        )
        self.dataset = self.dataset.cast_column("audio", Audio())
        self.data_loaded = True


class MockAudioClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        type="AudioClassification",
        name="MockAudioClassification",
        main_score="accuracy",
        **general_args,  # type: ignore[arg-type]
    )
    metadata.modalities = ["audio"]
    input_column_name = "audio"
    expected_stats = {
        "test": {
            "num_samples": 2,
            "samples_in_train": 2,
            "text_statistics": None,
            "image_statistics": None,
            "audio_statistics": {
                "total_duration_seconds": 3.0,
                "min_duration_seconds": 1.0,
                "average_duration_seconds": 1.5,
                "max_duration_seconds": 2.0,
                "unique_audios": 2,
                "average_sampling_rate": 12000.0,
                "sampling_rates": {16000: 1, 8000: 1},
            },
            "video_statistics": None,
            "label_statistics": {
                "min_labels_per_text": 1,
                "average_label_per_text": 1.0,
                "max_labels_per_text": 1,
                "unique_labels": 2,
                "labels": {"1": {"count": 1}, "2": {"count": 1}},
            },
        },
        "train": {
            "num_samples": 10,
            "samples_in_train": None,
            "text_statistics": None,
            "image_statistics": None,
            "audio_statistics": {
                "total_duration_seconds": 15.0,
                "min_duration_seconds": 1.0,
                "average_duration_seconds": 1.5,
                "max_duration_seconds": 2.0,
                "unique_audios": 2,
                "average_sampling_rate": 12000.0,
                "sampling_rates": {16000: 5, 8000: 5},
            },
            "video_statistics": None,
            "label_statistics": {
                "min_labels_per_text": 1,
                "average_label_per_text": 1.0,
                "max_labels_per_text": 1,
                "unique_labels": 2,
                "labels": {"1": {"count": 5}, "2": {"count": 5}},
            },
        },
    }

    def load_data(self, **kwargs):
        sampling_rates = [16000, 8000]
        mock_audio = [
            {
                "array": self.np_rng.random(16000),  # 1s
                "sampling_rate": sampling_rates[i],
            }
            for i in range(2)
        ]

        self.dataset = DatasetDict(
            {
                "test": Dataset.from_dict(
                    {
                        "audio": mock_audio,
                        "label": [1, 2],
                    }
                ),
                "train": Dataset.from_dict(
                    {
                        "audio": mock_audio * 5,
                        "label": [1, 2] * 5,
                    }
                ),
            }
        )
        self.dataset = self.dataset.cast_column("audio", Audio())

        self.data_loaded = True


class MockAudioClassificationCrossVal(AbsTaskClassification):
    metadata = TaskMetadata(
        type="AudioClassification",
        name="MockAudioClassificationCrossVal",
        main_score="accuracy",
        **general_args,  # type: ignore[arg-type]
    )
    metadata.modalities = ["audio"]
    metadata.eval_splits = ["train"]
    input_column_name = "audio"
    is_cross_validation = True
    expected_stats = {
        "train": {
            "num_samples": 10,
            "samples_in_train": None,
            "text_statistics": None,
            "image_statistics": None,
            "audio_statistics": {
                "total_duration_seconds": 10.0,
                "min_duration_seconds": 1.0,
                "average_duration_seconds": 1.0,
                "max_duration_seconds": 1.0,
                "unique_audios": 2,
                "average_sampling_rate": 16000.0,
                "sampling_rates": {16000: 10},
            },
            "video_statistics": None,
            "label_statistics": {
                "min_labels_per_text": 1,
                "average_label_per_text": 1.0,
                "max_labels_per_text": 1,
                "unique_labels": 2,
                "labels": {"1": {"count": 5}, "2": {"count": 5}},
            },
        }
    }

    def load_data(self, **kwargs):
        mock_audio = create_mock_audio(self.np_rng)

        self.dataset = DatasetDict(
            {
                "train": Dataset.from_dict(
                    {
                        "audio": mock_audio * 5,
                        "label": [1, 2] * 5,
                    }
                ),
            }
        )
        self.dataset = self.dataset.cast_column("audio", Audio())

        self.data_loaded = True


class MockVideoClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        type="VideoClassification",
        name="MockVideoClassification",
        main_score="accuracy",
        **general_args,  # type: ignore[arg-type]
    )
    metadata.modalities = ["video"]
    metadata.category = "v2c"
    input_column_name = "video"
    expected_stats = {
        "test": {
            "num_samples": 2,
            "samples_in_train": 2,
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
                "labels": {"1": {"count": 1}, "2": {"count": 1}},
            },
        },
        "train": {
            "num_samples": 10,
            "samples_in_train": None,
            "text_statistics": None,
            "image_statistics": None,
            "audio_statistics": None,
            "video_statistics": {
                "total_duration_seconds": 10.0,
                "total_frames": 240,
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
                "fps": {24: 10},
                "min_resolution": (64, 64),
                "average_resolution": (64.0, 64.0),
                "max_resolution": (64, 64),
                "resolutions": {"64x64": 10},
            },
            "label_statistics": {
                "min_labels_per_text": 1,
                "average_label_per_text": 1.0,
                "max_labels_per_text": 1,
                "unique_labels": 2,
                "labels": {"1": {"count": 5}, "2": {"count": 5}},
            },
        },
    }

    def load_data(self, **kwargs):
        from datasets import Video

        mock_videos = create_mock_video_bytes(self.np_rng)

        self.dataset = DatasetDict(
            {
                "test": Dataset.from_dict(
                    {
                        "video": mock_videos,
                        "label": [1, 2],
                    }
                ),
                "train": Dataset.from_dict(
                    {
                        "video": mock_videos * 5,
                        "label": [1, 2] * 5,
                    }
                ),
            }
        )

        self.dataset = self.dataset.cast_column("video", Video())
        self.data_loaded = True


class MockVideoAudioClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        type="VideoClassification",
        name="MockVideoAudioClassification",
        main_score="accuracy",
        **general_args,  # type: ignore[arg-type]
    )
    metadata.modalities = ["video", "audio"]
    metadata.category = "va2c"
    input_column_name = ("video", "audio")
    expected_stats = {
        "test": {
            "num_samples": 2,
            "samples_in_train": 2,
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
                "labels": {"1": {"count": 1}, "2": {"count": 1}},
            },
        },
        "train": {
            "num_samples": 10,
            "samples_in_train": None,
            "text_statistics": None,
            "image_statistics": None,
            "audio_statistics": {
                "total_duration_seconds": 10.0,
                "min_duration_seconds": 1.0,
                "average_duration_seconds": 1.0,
                "max_duration_seconds": 1.0,
                "unique_audios": 2,
                "average_sampling_rate": 16000.0,
                "sampling_rates": {16000: 10},
            },
            "video_statistics": {
                "total_duration_seconds": 10.0,
                "total_frames": 240,
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
                "fps": {24: 10},
                "min_resolution": (64, 64),
                "average_resolution": (64.0, 64.0),
                "max_resolution": (64, 64),
                "resolutions": {"64x64": 10},
            },
            "label_statistics": {
                "min_labels_per_text": 1,
                "average_label_per_text": 1.0,
                "max_labels_per_text": 1,
                "unique_labels": 2,
                "labels": {"1": {"count": 5}, "2": {"count": 5}},
            },
        },
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
                        "label": [1, 2],
                    }
                ),
                "train": Dataset.from_dict(
                    {
                        "video": mock_videos * 5,
                        "audio": mock_audio * 5,
                        "label": [1, 2] * 5,
                    }
                ),
            }
        )
        self.dataset = self.dataset.cast_column("video", Video())
        self.dataset = self.dataset.cast_column("audio", Audio())
        self.data_loaded = True


class MockVideoMultilabelClassificationTask(AbsTaskMultilabelClassification):
    expected_stats = {
        "test": {
            "num_samples": 2,
            "samples_in_train": 2,
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
        },
        "train": {
            "num_samples": 10,
            "samples_in_train": None,
            "text_statistics": None,
            "image_statistics": None,
            "audio_statistics": None,
            "video_statistics": {
                "total_duration_seconds": 10.0,
                "total_frames": 240,
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
                "fps": {24: 10},
                "min_resolution": (64, 64),
                "average_resolution": (64.0, 64.0),
                "max_resolution": (64, 64),
                "resolutions": {"64x64": 10},
            },
            "label_statistics": {
                "min_labels_per_text": 1,
                "average_label_per_text": 1.0,
                "max_labels_per_text": 1,
                "unique_labels": 2,
                "labels": {"0": {"count": 5}, "1": {"count": 5}},
            },
        },
    }

    metadata = TaskMetadata(
        type="VideoMultilabelClassification",
        name="MockVideoMultilabelClassification",
        main_score="accuracy",
        **general_args,
    )
    metadata.modalities = ["video"]
    metadata.category = "v2c"
    input_column_name = "video"

    def load_data(self, **kwargs):
        from datasets import Video

        mock_videos = create_mock_video_bytes(self.np_rng)
        labels = [[0], [1]]

        self.dataset = DatasetDict(
            {
                "test": Dataset.from_dict({"video": mock_videos, "label": labels}),
                "train": Dataset.from_dict(
                    {"video": mock_videos * 5, "label": labels * 5}
                ),
            }
        )

        self.dataset = self.dataset.cast_column("video", Video())
        self.data_loaded = True


class MockVideoAudioMultilabelClassificationTask(AbsTaskMultilabelClassification):
    expected_stats = {
        "test": {
            "num_samples": 2,
            "samples_in_train": 2,
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
        },
        "train": {
            "num_samples": 10,
            "samples_in_train": None,
            "text_statistics": None,
            "image_statistics": None,
            "audio_statistics": {
                "total_duration_seconds": 10.0,
                "min_duration_seconds": 1.0,
                "average_duration_seconds": 1.0,
                "max_duration_seconds": 1.0,
                "unique_audios": 2,
                "average_sampling_rate": 16000.0,
                "sampling_rates": {16000: 10},
            },
            "video_statistics": {
                "total_duration_seconds": 10.0,
                "total_frames": 240,
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
                "fps": {24: 10},
                "min_resolution": (64, 64),
                "average_resolution": (64.0, 64.0),
                "max_resolution": (64, 64),
                "resolutions": {"64x64": 10},
            },
            "label_statistics": {
                "min_labels_per_text": 1,
                "average_label_per_text": 1.0,
                "max_labels_per_text": 1,
                "unique_labels": 2,
                "labels": {"0": {"count": 5}, "1": {"count": 5}},
            },
        },
    }

    metadata = TaskMetadata(
        type="VideoMultilabelClassification",
        name="MockVideoAudioMultilabelClassification",
        main_score="accuracy",
        **general_args,
    )
    metadata.modalities = ["video", "audio"]
    metadata.category = "va2c"
    input_column_name = ("video", "audio")

    def load_data(self, **kwargs):
        from datasets import Video

        mock_videos = create_mock_video_bytes(self.np_rng)
        mock_audio = create_mock_audio(self.np_rng)
        labels = [[0], [1]]

        self.dataset = DatasetDict(
            {
                "test": Dataset.from_dict(
                    {"video": mock_videos, "audio": mock_audio, "label": labels}
                ),
                "train": Dataset.from_dict(
                    {
                        "video": mock_videos * 5,
                        "audio": mock_audio * 5,
                        "label": labels * 5,
                    }
                ),
            }
        )
        self.dataset = self.dataset.cast_column("video", Video())
        self.dataset = self.dataset.cast_column("audio", Audio())
        self.data_loaded = True
