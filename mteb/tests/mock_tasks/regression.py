from __future__ import annotations

from typing import TYPE_CHECKING

from datasets import Dataset, DatasetDict

from mteb.abstasks.regression import AbsTaskRegression
from mteb.abstasks.task_metadata import TaskMetadata

if TYPE_CHECKING:
    pass

from .utils import (
    create_mock_images,
    general_args,
)


class MockRegressionTask(AbsTaskRegression):
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
            "values_statistics": {"min_score": 0.0, "avg_score": 0.5, "max_score": 1.0},
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
            "values_statistics": {"min_score": 0.0, "avg_score": 0.5, "max_score": 1.0},
        },
    }

    metadata = TaskMetadata(
        type="Regression",
        name="MockRegressionTask",
        main_score="kendalltau",
        **general_args,
    )

    def load_data(self, **kwargs):
        train_texts = ["This is a test sentence", "This is another train sentence"]
        test_texts = ["This is a test sentence", "This is another test sentence"]
        train_values = [1.0, 0.0]
        test_values = [1.0, 0.0]

        self.dataset = DatasetDict(
            {
                "test": Dataset.from_dict(
                    {
                        "text": test_texts,
                        "value": test_values,
                    }
                ),
                "train": Dataset.from_dict(
                    {
                        "text": train_texts,
                        "value": train_values,
                    }
                ),
            }
        )
        self.data_loaded = True


class MockImageRegressionTask(AbsTaskRegression):
    expected_stats = {
        "test": {
            "num_samples": 2,
            "samples_in_train": 0,
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
            "values_statistics": {"min_score": 0.0, "avg_score": 0.5, "max_score": 1.0},
        },
        "train": {
            "num_samples": 2,
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
            "values_statistics": {"min_score": 0.0, "avg_score": 0.5, "max_score": 1.0},
        },
    }

    metadata = TaskMetadata(
        type="Regression",
        name="MockImageRegressionTask",
        main_score="kendalltau",
        **general_args,
    )
    metadata.modalities = ["image"]
    metadata.category = "i2c"
    input_column_name = "image"

    def load_data(self, **kwargs):
        train_images = create_mock_images(self.np_rng)
        test_images = create_mock_images(self.np_rng)

        train_values = [1.0, 0.0]
        test_values = [1.0, 0.0]

        self.dataset = DatasetDict(
            {
                "test": Dataset.from_dict(
                    {
                        "image": test_images,
                        "value": test_values,
                    }
                ),
                "train": Dataset.from_dict(
                    {
                        "image": train_images,
                        "value": train_values,
                    }
                ),
            }
        )
        self.data_loaded = True
