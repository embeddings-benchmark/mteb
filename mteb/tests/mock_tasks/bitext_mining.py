from __future__ import annotations

from typing import TYPE_CHECKING

from datasets import Dataset, DatasetDict

from mteb.abstasks.task_metadata import TaskMetadata
from mteb.abstasks.text.bitext_mining import AbsTaskBitextMining

if TYPE_CHECKING:
    pass

from .utils import (
    general_args,
    multilingual_eval_langs,
)


class MockBitextMiningTask(AbsTaskBitextMining):
    expected_stats = {
        "test": {
            "num_samples": 2,
            "number_of_characters": 113,
            "unique_pairs": 2,
            "sentence1_statistics": {
                "total_text_length": 52,
                "min_text_length": 23,
                "average_text_length": 26.0,
                "max_text_length": 29,
                "unique_texts": 2,
            },
            "sentence2_statistics": {
                "total_text_length": 61,
                "min_text_length": 24,
                "average_text_length": 30.5,
                "max_text_length": 37,
                "unique_texts": 2,
            },
        }
    }

    metadata = TaskMetadata(
        type="BitextMining",
        name="MockBitextMiningTask",
        main_score="accuracy",
        **general_args,
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        sentence1 = ["This is a test sentence", "This is another test sentence"]
        sentence2 = [
            "dette er en test sætning",
            "denne her matche ikke den ovenstående",
        ]  # "this is a test sentence", "this does not match the above"

        self.dataset = DatasetDict(
            {
                "test": Dataset.from_dict(
                    {
                        "sentence1": sentence1,
                        "sentence2": sentence2,
                    }
                ),
            }
        )
        self.data_loaded = True


class MockMultilingualBitextMiningTask(AbsTaskBitextMining):
    expected_stats = {
        "test": {
            "num_samples": 4,
            "number_of_characters": 226,
            "unique_pairs": 2,
            "sentence1_statistics": {
                "total_text_length": 104,
                "min_text_length": 23,
                "average_text_length": 26.0,
                "max_text_length": 29,
                "unique_texts": 2,
            },
            "sentence2_statistics": {
                "total_text_length": 122,
                "min_text_length": 24,
                "average_text_length": 30.5,
                "max_text_length": 37,
                "unique_texts": 2,
            },
            "hf_subset_descriptive_stats": {
                "eng": {
                    "num_samples": 2,
                    "number_of_characters": 113,
                    "unique_pairs": 2,
                    "sentence1_statistics": {
                        "total_text_length": 52,
                        "min_text_length": 23,
                        "average_text_length": 26.0,
                        "max_text_length": 29,
                        "unique_texts": 2,
                    },
                    "sentence2_statistics": {
                        "total_text_length": 61,
                        "min_text_length": 24,
                        "average_text_length": 30.5,
                        "max_text_length": 37,
                        "unique_texts": 2,
                    },
                },
                "fra": {
                    "num_samples": 2,
                    "number_of_characters": 113,
                    "unique_pairs": 2,
                    "sentence1_statistics": {
                        "total_text_length": 52,
                        "min_text_length": 23,
                        "average_text_length": 26.0,
                        "max_text_length": 29,
                        "unique_texts": 2,
                    },
                    "sentence2_statistics": {
                        "total_text_length": 61,
                        "min_text_length": 24,
                        "average_text_length": 30.5,
                        "max_text_length": 37,
                        "unique_texts": 2,
                    },
                },
            },
        }
    }

    metadata = TaskMetadata(
        type="BitextMining",
        name="MockMultilingualBitextMiningTask",
        main_score="accuracy",
        **general_args,
    )
    metadata.eval_langs = multilingual_eval_langs

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        sentence1 = ["This is a test sentence", "This is another test sentence"]
        sentence2 = [
            "dette er en test sætning",
            "denne her matche ikke den ovenstående",
        ]  # "this is a test sentence", "this does not match the above"
        data = {
            "test": Dataset.from_dict(
                {
                    "sentence1": sentence1,
                    "sentence2": sentence2,
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


class MockMultilingualParallelBitextMiningTask(AbsTaskBitextMining):
    parallel_subsets = True
    expected_stats = {
        "test": {
            "num_samples": 4,
            "number_of_characters": 226,
            "unique_pairs": 4,
            "sentence1_statistics": {
                "total_text_length": 113,
                "min_text_length": 23,
                "average_text_length": 28.25,
                "max_text_length": 37,
                "unique_texts": 4,
            },
            "sentence2_statistics": {
                "total_text_length": 113,
                "min_text_length": 23,
                "average_text_length": 28.25,
                "max_text_length": 37,
                "unique_texts": 4,
            },
            "hf_subset_descriptive_stats": {
                "eng_Latn-fra_Latn": {
                    "num_samples": 2,
                    "number_of_characters": 113,
                    "unique_pairs": 2,
                    "sentence1_statistics": {
                        "total_text_length": 52,
                        "min_text_length": 23,
                        "average_text_length": 26.0,
                        "max_text_length": 29,
                        "unique_texts": 2,
                    },
                    "sentence2_statistics": {
                        "total_text_length": 61,
                        "min_text_length": 24,
                        "average_text_length": 30.5,
                        "max_text_length": 37,
                        "unique_texts": 2,
                    },
                },
                "fra_Latn-eng_Latn": {
                    "num_samples": 2,
                    "number_of_characters": 113,
                    "unique_pairs": 2,
                    "sentence1_statistics": {
                        "total_text_length": 61,
                        "min_text_length": 24,
                        "average_text_length": 30.5,
                        "max_text_length": 37,
                        "unique_texts": 2,
                    },
                    "sentence2_statistics": {
                        "total_text_length": 52,
                        "min_text_length": 23,
                        "average_text_length": 26.0,
                        "max_text_length": 29,
                        "unique_texts": 2,
                    },
                },
            },
        }
    }

    metadata = TaskMetadata(
        type="BitextMining",
        name="MockMultilingualParallelBitextMiningTask",
        main_score="accuracy",
        **general_args,
    )
    metadata.eval_langs = {
        "eng_Latn-fra_Latn": ["eng-Latn", "fra-Latn"],
        "fra_Latn-eng_Latn": ["eng-Latn", "fra-Latn"],
    }

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        sentence1 = ["This is a test sentence", "This is another test sentence"]
        sentence2 = [
            "dette er en test sætning",
            "denne her matche ikke den ovenstående",
        ]  # "this is a test sentence", "this does not match the above"

        self.dataset = DatasetDict(
            {
                "test": Dataset.from_dict(
                    {
                        "eng_Latn": sentence1,
                        "fra_Latn": sentence2,
                    }
                ),
            }
        )
        self.data_loaded = True
