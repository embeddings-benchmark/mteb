from __future__ import annotations

from datasets import Dataset, DatasetDict

from mteb.abstasks.task_metadata import TaskMetadata
from mteb.abstasks.text.summarization import AbsTaskSummarization

from .utils import (
    general_args,
    multilingual_eval_langs,
)


class MockSummarizationTask(AbsTaskSummarization):
    expected_stats = {
        "test": {
            "num_samples": 2,
            "number_of_characters": 244,
            "text_statistics": {
                "total_text_length": 52,
                "min_text_length": 23,
                "average_text_length": 26.0,
                "max_text_length": 29,
                "unique_texts": 2,
            },
            "human_summaries_statistics": {
                "total_text_length": 80,
                "min_text_length": 17,
                "average_text_length": 20.0,
                "max_text_length": 23,
                "unique_texts": 2,
            },
            "machine_summaries_statistics": {
                "total_text_length": 112,
                "min_text_length": 25,
                "average_text_length": 28.0,
                "max_text_length": 31,
                "unique_texts": 2,
            },
            "score_statistics": {"min_score": 0, "avg_score": 0.5, "max_score": 1},
        }
    }

    metadata = TaskMetadata(
        type="Summarization",
        name="MockSummarizationTask",
        main_score="cosine_spearman",
        **general_args,
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        texts = ["This is a test sentence", "This is another test sentence"]
        human_summaries = [
            ["This is a summary", "This is another summary"],
            ["This is a summary", "This is another summary"],
        ]
        machine_summaries = [
            ["This is a machine summary", "This is another machine summary"],
            ["This is a machine summary", "This is another machine summary"],
        ]
        relevance = [[1, 0], [0, 1]]

        self.dataset = DatasetDict(
            {
                "test": Dataset.from_dict(
                    {
                        "text": texts,
                        "human_summaries": human_summaries,
                        "machine_summaries": machine_summaries,
                        "relevance": relevance,
                    }
                ),
            }
        )
        self.data_loaded = True

    min_score = 0
    max_score = 1


class MockMultilingualSummarizationTask(AbsTaskSummarization):
    expected_stats = {
        "test": {
            "num_samples": 4,
            "number_of_characters": 488,
            "text_statistics": {
                "total_text_length": 104,
                "min_text_length": 23,
                "average_text_length": 26.0,
                "max_text_length": 29,
                "unique_texts": 2,
            },
            "human_summaries_statistics": {
                "total_text_length": 160,
                "min_text_length": 17,
                "average_text_length": 20.0,
                "max_text_length": 23,
                "unique_texts": 2,
            },
            "machine_summaries_statistics": {
                "total_text_length": 224,
                "min_text_length": 25,
                "average_text_length": 28.0,
                "max_text_length": 31,
                "unique_texts": 2,
            },
            "score_statistics": {"min_score": 0, "avg_score": 0.5, "max_score": 1},
            "hf_subset_descriptive_stats": {
                "eng": {
                    "num_samples": 2,
                    "number_of_characters": 244,
                    "text_statistics": {
                        "total_text_length": 52,
                        "min_text_length": 23,
                        "average_text_length": 26.0,
                        "max_text_length": 29,
                        "unique_texts": 2,
                    },
                    "human_summaries_statistics": {
                        "total_text_length": 80,
                        "min_text_length": 17,
                        "average_text_length": 20.0,
                        "max_text_length": 23,
                        "unique_texts": 2,
                    },
                    "machine_summaries_statistics": {
                        "total_text_length": 112,
                        "min_text_length": 25,
                        "average_text_length": 28.0,
                        "max_text_length": 31,
                        "unique_texts": 2,
                    },
                    "score_statistics": {
                        "min_score": 0,
                        "avg_score": 0.5,
                        "max_score": 1,
                    },
                },
                "fra": {
                    "num_samples": 2,
                    "number_of_characters": 244,
                    "text_statistics": {
                        "total_text_length": 52,
                        "min_text_length": 23,
                        "average_text_length": 26.0,
                        "max_text_length": 29,
                        "unique_texts": 2,
                    },
                    "human_summaries_statistics": {
                        "total_text_length": 80,
                        "min_text_length": 17,
                        "average_text_length": 20.0,
                        "max_text_length": 23,
                        "unique_texts": 2,
                    },
                    "machine_summaries_statistics": {
                        "total_text_length": 112,
                        "min_text_length": 25,
                        "average_text_length": 28.0,
                        "max_text_length": 31,
                        "unique_texts": 2,
                    },
                    "score_statistics": {
                        "min_score": 0,
                        "avg_score": 0.5,
                        "max_score": 1,
                    },
                },
            },
        }
    }

    metadata = TaskMetadata(
        type="Summarization",
        name="MockMultilingualSummarizationTask",
        main_score="cosine_spearman",
        **general_args,
    )
    metadata.eval_langs = multilingual_eval_langs

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        texts = ["This is a test sentence", "This is another test sentence"]
        human_summaries = [
            ["This is a summary", "This is another summary"],
            ["This is a summary", "This is another summary"],
        ]
        machine_summaries = [
            ["This is a machine summary", "This is another machine summary"],
            ["This is a machine summary", "This is another machine summary"],
        ]
        relevance = [[1, 0], [0, 1]]
        data = {
            "test": Dataset.from_dict(
                {
                    "text": texts,
                    "human_summaries": human_summaries,
                    "machine_summaries": machine_summaries,
                    "relevance": relevance,
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

    min_score = 0
    max_score = 1
