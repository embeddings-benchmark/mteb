"""This implements minimal viable mock tasks for testing the benchmarking framework."""

from __future__ import annotations

import numpy as np
from datasets import Dataset, DatasetDict
from PIL import Image

from mteb.abstasks import MultilingualTask
from mteb.abstasks.AbsTaskBitextMining import AbsTaskBitextMining
from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.AbsTaskClustering import AbsTaskClustering
from mteb.abstasks.AbsTaskClusteringFast import AbsTaskClusteringFast
from mteb.abstasks.AbsTaskInstructionRetrieval import AbsTaskInstructionRetrieval
from mteb.abstasks.AbsTaskMultilabelClassification import (
    AbsTaskMultilabelClassification,
)
from mteb.abstasks.AbsTaskPairClassification import AbsTaskPairClassification
from mteb.abstasks.AbsTaskReranking import AbsTaskReranking
from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.AbsTaskSTS import AbsTaskSTS
from mteb.abstasks.AbsTaskSummarization import AbsTaskSummarization
from mteb.abstasks.Image.AbsTaskAny2AnyMultiChoice import AbsTaskAny2AnyMultiChoice
from mteb.abstasks.Image.AbsTaskAny2AnyRetrieval import AbsTaskAny2AnyRetrieval
from mteb.abstasks.Image.AbsTaskImageClassification import AbsTaskImageClassification
from mteb.abstasks.Image.AbsTaskImageClustering import AbsTaskImageClustering
from mteb.abstasks.Image.AbsTaskImageMultilabelClassification import (  # noqa
    AbsTaskImageMultilabelClassification,
)
from mteb.abstasks.Image.AbsTaskImageTextPairClassification import (
    AbsTaskImageTextPairClassification,
)
from mteb.abstasks.Image.AbsTaskVisualSTS import AbsTaskVisualSTS
from mteb.abstasks.Image.AbsTaskZeroShotClassification import (
    AbsTaskZeroShotClassification,
)
from mteb.abstasks.TaskMetadata import TaskMetadata

general_args = {
    "description": "a mock task for testing",
    "reference": "https://github.com/embeddings-benchmark/mteb",
    "dataset": {
        "path": "NA",
        "revision": "NA",
    },
    "category": "s2s",
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


class MockClassificationTask(AbsTaskClassification):
    expected_stats = {
        "test": {
            "num_samples": 2,
            "number_of_characters": 52,
            "num_texts_in_train": 1,
            "min_text_length": 23,
            "average_text_length": 26.0,
            "max_text_length": 29,
            "unique_text": 2,
            "unique_labels": 2,
            "labels": {"0": {"count": 1}, "1": {"count": 1}},
        },
        "train": {
            "num_samples": 2,
            "number_of_characters": 53,
            "num_texts_in_train": None,
            "min_text_length": 23,
            "average_text_length": 26.5,
            "max_text_length": 30,
            "unique_text": 2,
            "unique_labels": 2,
            "labels": {"0": {"count": 1}, "1": {"count": 1}},
        },
    }

    metadata = TaskMetadata(
        type="Classification",
        name="MockClassificationTask",
        main_score="accuracy",
        **general_args,  # type: ignore
    )

    def load_data(self, **kwargs):
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


class MockMultilingualClassificationTask(AbsTaskClassification, MultilingualTask):
    expected_stats = {
        "test": {
            "num_samples": 4,
            "number_of_characters": 104,
            "num_texts_in_train": 1,
            "min_text_length": 23,
            "average_text_length": 26.0,
            "max_text_length": 29,
            "unique_text": 2,
            "unique_labels": 2,
            "labels": {"0": {"count": 2}, "1": {"count": 2}},
            "hf_subset_descriptive_stats": {
                "eng": {
                    "num_samples": 2,
                    "number_of_characters": 52,
                    "num_texts_in_train": 1,
                    "min_text_length": 23,
                    "average_text_length": 26.0,
                    "max_text_length": 29,
                    "unique_text": 2,
                    "unique_labels": 2,
                    "labels": {"0": {"count": 1}, "1": {"count": 1}},
                },
                "fra": {
                    "num_samples": 2,
                    "number_of_characters": 52,
                    "num_texts_in_train": 1,
                    "min_text_length": 23,
                    "average_text_length": 26.0,
                    "max_text_length": 29,
                    "unique_text": 2,
                    "unique_labels": 2,
                    "labels": {"0": {"count": 1}, "1": {"count": 1}},
                },
            },
        },
        "train": {
            "num_samples": 4,
            "number_of_characters": 106,
            "num_texts_in_train": None,
            "min_text_length": 23,
            "average_text_length": 26.5,
            "max_text_length": 30,
            "unique_text": 2,
            "unique_labels": 2,
            "labels": {"0": {"count": 2}, "1": {"count": 2}},
            "hf_subset_descriptive_stats": {
                "eng": {
                    "num_samples": 2,
                    "number_of_characters": 53,
                    "num_texts_in_train": None,
                    "min_text_length": 23,
                    "average_text_length": 26.5,
                    "max_text_length": 30,
                    "unique_text": 2,
                    "unique_labels": 2,
                    "labels": {"0": {"count": 1}, "1": {"count": 1}},
                },
                "fra": {
                    "num_samples": 2,
                    "number_of_characters": 53,
                    "num_texts_in_train": None,
                    "min_text_length": 23,
                    "average_text_length": 26.5,
                    "max_text_length": 30,
                    "unique_text": 2,
                    "unique_labels": 2,
                    "labels": {"0": {"count": 1}, "1": {"count": 1}},
                },
            },
        },
    }

    metadata = TaskMetadata(
        type="Classification",
        name="MockMultilingualClassificationTask",
        main_score="accuracy",
        **general_args,  # type: ignore
    )
    metadata.eval_langs = multilingual_eval_langs

    def load_data(self, **kwargs):
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


class MockBitextMiningTask(AbsTaskBitextMining):
    expected_stats = {
        "test": {
            "num_samples": 2,
            "number_of_characters": 113,
            "unique_pairs": 2,
            "min_sentence1_length": 23,
            "average_sentence1_length": 26.0,
            "max_sentence1_length": 29,
            "unique_sentence1": 2,
            "min_sentence2_length": 24,
            "average_sentence2_length": 30.5,
            "max_sentence2_length": 37,
            "unique_sentence2": 2,
        }
    }

    metadata = TaskMetadata(
        type="BitextMining",
        name="MockBitextMiningTask",
        main_score="accuracy",
        **general_args,  # type: ignore
    )

    def load_data(self, **kwargs):
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


class MockMultilingualBitextMiningTask(AbsTaskBitextMining, MultilingualTask):
    expected_stats = {
        "test": {
            "num_samples": 4,
            "number_of_characters": 226,
            "unique_pairs": 2,
            "min_sentence1_length": 23,
            "average_sentence1_length": 26.0,
            "max_sentence1_length": 29,
            "unique_sentence1": 2,
            "min_sentence2_length": 24,
            "average_sentence2_length": 30.5,
            "max_sentence2_length": 37,
            "unique_sentence2": 2,
            "hf_subset_descriptive_stats": {
                "eng": {
                    "num_samples": 2,
                    "number_of_characters": 113,
                    "unique_pairs": 2,
                    "min_sentence1_length": 23,
                    "average_sentence1_length": 26.0,
                    "max_sentence1_length": 29,
                    "unique_sentence1": 2,
                    "min_sentence2_length": 24,
                    "average_sentence2_length": 30.5,
                    "max_sentence2_length": 37,
                    "unique_sentence2": 2,
                },
                "fra": {
                    "num_samples": 2,
                    "number_of_characters": 113,
                    "unique_pairs": 2,
                    "min_sentence1_length": 23,
                    "average_sentence1_length": 26.0,
                    "max_sentence1_length": 29,
                    "unique_sentence1": 2,
                    "min_sentence2_length": 24,
                    "average_sentence2_length": 30.5,
                    "max_sentence2_length": 37,
                    "unique_sentence2": 2,
                },
            },
        }
    }

    metadata = TaskMetadata(
        type="BitextMining",
        name="MockMultilingualBitextMiningTask",
        main_score="accuracy",
        **general_args,  # type: ignore
    )
    metadata.eval_langs = multilingual_eval_langs

    def load_data(self, **kwargs):
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


class MockMultilingualParallelBitextMiningTask(AbsTaskBitextMining, MultilingualTask):
    parallel_subsets = True
    expected_stats = {
        "test": {
            "num_samples": 4,
            "number_of_characters": 226,
            "unique_pairs": 4,
            "min_sentence1_length": 23,
            "average_sentence1_length": 28.25,
            "max_sentence1_length": 37,
            "unique_sentence1": 4,
            "min_sentence2_length": 23,
            "average_sentence2_length": 28.25,
            "max_sentence2_length": 37,
            "unique_sentence2": 4,
            "hf_subset_descriptive_stats": {
                "eng_Latn-fra_Latn": {
                    "num_samples": 2,
                    "number_of_characters": 113,
                    "unique_pairs": 2,
                    "min_sentence1_length": 23,
                    "average_sentence1_length": 26.0,
                    "max_sentence1_length": 29,
                    "unique_sentence1": 2,
                    "min_sentence2_length": 24,
                    "average_sentence2_length": 30.5,
                    "max_sentence2_length": 37,
                    "unique_sentence2": 2,
                },
                "fra_Latn-eng_Latn": {
                    "num_samples": 2,
                    "number_of_characters": 113,
                    "unique_pairs": 2,
                    "min_sentence1_length": 24,
                    "average_sentence1_length": 30.5,
                    "max_sentence1_length": 37,
                    "unique_sentence1": 2,
                    "min_sentence2_length": 23,
                    "average_sentence2_length": 26.0,
                    "max_sentence2_length": 29,
                    "unique_sentence2": 2,
                },
            },
        }
    }

    metadata = TaskMetadata(
        type="BitextMining",
        name="MockMultilingualParallelBitextMiningTask",
        main_score="accuracy",
        **general_args,  # type: ignore
    )
    metadata.eval_langs = {
        "eng_Latn-fra_Latn": ["eng-Latn", "fra-Latn"],
        "fra_Latn-eng_Latn": ["eng-Latn", "fra-Latn"],
    }

    def load_data(self, **kwargs):
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


class MockClusteringTask(AbsTaskClustering):
    expected_stats = {
        "test": {
            "num_samples": 1,
            "number_of_characters": 3,
            "min_text_length": 3,
            "average_text_length": 3.0,
            "max_text_length": 3,
            "unique_texts": 3,
            "min_labels_per_text": 1,
            "average_labels_per_text": 3.0,
            "max_labels_per_text": 1,
            "unique_labels": 3,
            "labels": {"0": {"count": 1}, "1": {"count": 1}, "2": {"count": 1}},
        }
    }

    metadata = TaskMetadata(
        type="Clustering",
        name="MockClusteringTask",
        main_score="v_measure",
        **general_args,  # type: ignore
    )

    def load_data(self, **kwargs):
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


class MockMultilingualClusteringTask(AbsTaskClustering, MultilingualTask):
    expected_stats = {
        "test": {
            "num_samples": 2,
            "number_of_characters": 6,
            "min_text_length": 3,
            "average_text_length": 3.0,
            "max_text_length": 3,
            "unique_texts": 3,
            "min_labels_per_text": 2,
            "average_labels_per_text": 3.0,
            "max_labels_per_text": 2,
            "unique_labels": 3,
            "labels": {"0": {"count": 2}, "1": {"count": 2}, "2": {"count": 2}},
            "hf_subset_descriptive_stats": {
                "eng": {
                    "num_samples": 1,
                    "number_of_characters": 3,
                    "min_text_length": 3,
                    "average_text_length": 3.0,
                    "max_text_length": 3,
                    "unique_texts": 3,
                    "min_labels_per_text": 1,
                    "average_labels_per_text": 3.0,
                    "max_labels_per_text": 1,
                    "unique_labels": 3,
                    "labels": {"0": {"count": 1}, "1": {"count": 1}, "2": {"count": 1}},
                },
                "fra": {
                    "num_samples": 1,
                    "number_of_characters": 3,
                    "min_text_length": 3,
                    "average_text_length": 3.0,
                    "max_text_length": 3,
                    "unique_texts": 3,
                    "min_labels_per_text": 1,
                    "average_labels_per_text": 3.0,
                    "max_labels_per_text": 1,
                    "unique_labels": 3,
                    "labels": {"0": {"count": 1}, "1": {"count": 1}, "2": {"count": 1}},
                },
            },
        }
    }

    metadata = TaskMetadata(
        type="Clustering",
        name="MockMultilingualClusteringTask",
        main_score="v_measure",
        **general_args,  # type: ignore
    )
    metadata.eval_langs = multilingual_eval_langs

    def load_data(self, **kwargs):
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


class MockClusteringFastTask(AbsTaskClusteringFast):
    max_document_to_embed = 3
    max_fraction_of_documents_to_embed = None
    expected_stats = {
        "test": {
            "num_samples": 3,
            "number_of_characters": 81,
            "min_text_length": 23,
            "average_text_length": 27.0,
            "max_text_length": 29,
            "min_labels_per_text": 1,
            "average_labels_per_text": 1.0,
            "max_labels_per_text": 1,
            "unique_labels": 3,
            "labels": {"0": {"count": 1}, "1": {"count": 1}, "2": {"count": 1}},
        }
    }

    metadata = TaskMetadata(
        type="Clustering",
        name="MockClusteringFastTask",
        main_score="v_measure",
        **general_args,  # type: ignore
    )

    def load_data(self, **kwargs):
        sentences = [
            "This is a test sentence",
            "This is another test sentence",
            "This is a third test sentence",
        ]
        labels = [0, 1, 2]

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


class MockMultilingualClusteringFastTask(AbsTaskClusteringFast, MultilingualTask):
    max_document_to_embed = 3
    max_fraction_of_documents_to_embed = None
    expected_stats = {
        "test": {
            "num_samples": 6,
            "number_of_characters": 162,
            "min_text_length": 23,
            "average_text_length": 27.0,
            "max_text_length": 29,
            "min_labels_per_text": 2,
            "average_labels_per_text": 1.0,
            "max_labels_per_text": 2,
            "unique_labels": 3,
            "labels": {"0": {"count": 2}, "1": {"count": 2}, "2": {"count": 2}},
            "hf_subset_descriptive_stats": {
                "eng": {
                    "num_samples": 3,
                    "number_of_characters": 81,
                    "min_text_length": 23,
                    "average_text_length": 27.0,
                    "max_text_length": 29,
                    "min_labels_per_text": 1,
                    "average_labels_per_text": 1.0,
                    "max_labels_per_text": 1,
                    "unique_labels": 3,
                    "labels": {"0": {"count": 1}, "1": {"count": 1}, "2": {"count": 1}},
                },
                "fra": {
                    "num_samples": 3,
                    "number_of_characters": 81,
                    "min_text_length": 23,
                    "average_text_length": 27.0,
                    "max_text_length": 29,
                    "min_labels_per_text": 1,
                    "average_labels_per_text": 1.0,
                    "max_labels_per_text": 1,
                    "unique_labels": 3,
                    "labels": {"0": {"count": 1}, "1": {"count": 1}, "2": {"count": 1}},
                },
            },
        }
    }

    metadata = TaskMetadata(
        type="Clustering",
        name="MockMultilingualClusteringFastTask",
        main_score="v_measure",
        **general_args,  # type: ignore
    )
    metadata.eval_langs = multilingual_eval_langs

    def load_data(self, **kwargs):
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


class MockPairClassificationTask(AbsTaskPairClassification):
    expected_stats = {
        "test": {
            "num_samples": 2,
            "number_of_characters": 113,
            "min_sentence1_length": 23,
            "avg_sentence1_length": 26.0,
            "max_sentence1_length": 29,
            "unique_sentence1": 2,
            "min_sentence2_length": 24,
            "avg_sentence2_length": 30.5,
            "max_sentence2_length": 37,
            "unique_sentence2": 2,
            "unique_labels": 2,
            "labels": {"1": {"count": 1}, "0": {"count": 1}},
        }
    }

    metadata = TaskMetadata(
        type="PairClassification",
        name="MockPairClassificationTask",
        main_score="similarity_ap",
        **general_args,  # type: ignore
    )

    def load_data(self, **kwargs):
        sentence1 = [["This is a test sentence", "This is another test sentence"]]
        sentence2 = [
            [
                "dette er en test sætning",
                "denne her matche ikke den ovenstående",
            ]
        ]  # "this is a test sentence", "this does not match the above"
        labels = [[1, 0]]

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


class MockMultilingualPairClassificationTask(
    AbsTaskPairClassification, MultilingualTask
):
    expected_stats = {
        "test": {
            "num_samples": 4,
            "number_of_characters": 226,
            "min_sentence1_length": 23,
            "avg_sentence1_length": 26.0,
            "max_sentence1_length": 29,
            "unique_sentence1": 2,
            "min_sentence2_length": 24,
            "avg_sentence2_length": 30.5,
            "max_sentence2_length": 37,
            "unique_sentence2": 2,
            "unique_labels": 2,
            "labels": {"1": {"count": 2}, "0": {"count": 2}},
            "hf_subset_descriptive_stats": {
                "eng": {
                    "num_samples": 2,
                    "number_of_characters": 113,
                    "min_sentence1_length": 23,
                    "avg_sentence1_length": 26.0,
                    "max_sentence1_length": 29,
                    "unique_sentence1": 2,
                    "min_sentence2_length": 24,
                    "avg_sentence2_length": 30.5,
                    "max_sentence2_length": 37,
                    "unique_sentence2": 2,
                    "unique_labels": 2,
                    "labels": {"1": {"count": 1}, "0": {"count": 1}},
                },
                "fra": {
                    "num_samples": 2,
                    "number_of_characters": 113,
                    "min_sentence1_length": 23,
                    "avg_sentence1_length": 26.0,
                    "max_sentence1_length": 29,
                    "unique_sentence1": 2,
                    "min_sentence2_length": 24,
                    "avg_sentence2_length": 30.5,
                    "max_sentence2_length": 37,
                    "unique_sentence2": 2,
                    "unique_labels": 2,
                    "labels": {"1": {"count": 1}, "0": {"count": 1}},
                },
            },
        }
    }

    metadata = TaskMetadata(
        type="PairClassification",
        name="MockMultilingualPairClassificationTask",
        main_score="similarity_ap",
        **general_args,  # type: ignore
    )
    metadata.eval_langs = multilingual_eval_langs

    def load_data(self, **kwargs):
        sentence1 = ["This is a test sentence", "This is another test sentence"]
        sentence2 = [
            "dette er en test sætning",
            "denne her matche ikke den ovenstående",
        ]
        # "this is a test sentence", "this does not match the above"
        labels = [1, 0]
        data = {
            "test": [
                {
                    "sentence1": sentence1,
                    "sentence2": sentence2,
                    "labels": labels,
                }
            ]
        }

        self.dataset = DatasetDict(
            {
                "eng": data,
                "fra": data,
            }
        )
        self.data_loaded = True


class MockSTSTask(AbsTaskSTS):
    expected_stats = {
        "test": {
            "num_samples": 2,
            "number_of_characters": 113,
            "min_sentence1_length": 23,
            "average_sentence1_len": 26.0,
            "max_sentence1_length": 29,
            "unique_sentence1": 2,
            "min_sentence2_length": 24,
            "average_sentence2_len": 30.5,
            "max_sentence2_length": 37,
            "unique_sentence2": 2,
            "min_score": 0,
            "avg_score": 0.5,
            "max_score": 1,
        }
    }

    metadata = TaskMetadata(
        type="STS",
        name="MockSTSTask",
        main_score="cosine_spearman",
        **general_args,  # type: ignore
    )

    def load_data(self, **kwargs):
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

    @property
    def metadata_dict(self) -> dict[str, str]:
        metadata_dict = super().metadata_dict
        metadata_dict["min_score"] = 0
        metadata_dict["max_score"] = 1
        return metadata_dict


class MockMultilingualSTSTask(AbsTaskSTS, MultilingualTask):
    expected_stats = {
        "test": {
            "num_samples": 4,
            "number_of_characters": 226,
            "min_sentence1_length": 23,
            "average_sentence1_len": 26.0,
            "max_sentence1_length": 29,
            "unique_sentence1": 2,
            "min_sentence2_length": 24,
            "average_sentence2_len": 30.5,
            "max_sentence2_length": 37,
            "unique_sentence2": 2,
            "min_score": 0,
            "avg_score": 0.5,
            "max_score": 1,
            "hf_subset_descriptive_stats": {
                "eng": {
                    "num_samples": 2,
                    "number_of_characters": 113,
                    "min_sentence1_length": 23,
                    "average_sentence1_len": 26.0,
                    "max_sentence1_length": 29,
                    "unique_sentence1": 2,
                    "min_sentence2_length": 24,
                    "average_sentence2_len": 30.5,
                    "max_sentence2_length": 37,
                    "unique_sentence2": 2,
                    "min_score": 0,
                    "avg_score": 0.5,
                    "max_score": 1,
                },
                "fra": {
                    "num_samples": 2,
                    "number_of_characters": 113,
                    "min_sentence1_length": 23,
                    "average_sentence1_len": 26.0,
                    "max_sentence1_length": 29,
                    "unique_sentence1": 2,
                    "min_sentence2_length": 24,
                    "average_sentence2_len": 30.5,
                    "max_sentence2_length": 37,
                    "unique_sentence2": 2,
                    "min_score": 0,
                    "avg_score": 0.5,
                    "max_score": 1,
                },
            },
        }
    }

    metadata = TaskMetadata(
        type="STS",
        name="MockMultilingualSTSTask",
        main_score="cosine_spearman",
        **general_args,  # type: ignore
    )
    metadata.eval_langs = multilingual_eval_langs

    def load_data(self, **kwargs):
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

    @property
    def metadata_dict(self) -> dict[str, str]:
        metadata_dict = super().metadata_dict
        metadata_dict["min_score"] = 0
        metadata_dict["max_score"] = 1
        return metadata_dict


class MockSummarizationTask(AbsTaskSummarization):
    expected_stats = {
        "test": {
            "num_samples": 2,
            "number_of_characters": 60,
            "min_text_length": 23,
            "avg_text_length": 26.0,
            "max_text_length": 29,
            "unique_texts": 2,
            "min_human_summaries_length": 2,
            "avg_human_summaries_length": 2.0,
            "max_human_summaries_length": 2,
            "unique_human_summaries": 2,
            "min_machine_summaries_length": 2,
            "avg_machine_summaries_length": 2.0,
            "max_machine_summaries_length": 2,
            "unique_machine_summaries": 2,
            "min_relevance": [0, 1],
            "avg_relevance": 0.5,
            "max_relevance": [1, 0],
        }
    }

    metadata = TaskMetadata(
        type="Summarization",
        name="MockSummarizationTask",
        main_score="cosine_spearman",
        **general_args,  # type: ignore
    )

    def load_data(self, **kwargs):
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

    @property
    def metadata_dict(self) -> dict[str, str]:
        metadata_dict = super().metadata_dict
        metadata_dict["min_score"] = 0
        metadata_dict["max_score"] = 1
        return metadata_dict


class MockMultilingualSummarizationTask(AbsTaskSummarization, MultilingualTask):
    expected_stats = {
        "test": {
            "num_samples": 4,
            "number_of_characters": 120,
            "min_text_length": 23,
            "avg_text_length": 26.0,
            "max_text_length": 29,
            "unique_texts": 2,
            "min_human_summaries_length": 2,
            "avg_human_summaries_length": 2.0,
            "max_human_summaries_length": 2,
            "unique_human_summaries": 2,
            "min_machine_summaries_length": 2,
            "avg_machine_summaries_length": 2.0,
            "max_machine_summaries_length": 2,
            "unique_machine_summaries": 2,
            "min_relevance": [0, 1],
            "avg_relevance": 0.5,
            "max_relevance": [1, 0],
            "hf_subset_descriptive_stats": {
                "eng": {
                    "num_samples": 2,
                    "number_of_characters": 60,
                    "min_text_length": 23,
                    "avg_text_length": 26.0,
                    "max_text_length": 29,
                    "unique_texts": 2,
                    "min_human_summaries_length": 2,
                    "avg_human_summaries_length": 2.0,
                    "max_human_summaries_length": 2,
                    "unique_human_summaries": 2,
                    "min_machine_summaries_length": 2,
                    "avg_machine_summaries_length": 2.0,
                    "max_machine_summaries_length": 2,
                    "unique_machine_summaries": 2,
                    "min_relevance": [0, 1],
                    "avg_relevance": 0.5,
                    "max_relevance": [1, 0],
                },
                "fra": {
                    "num_samples": 2,
                    "number_of_characters": 60,
                    "min_text_length": 23,
                    "avg_text_length": 26.0,
                    "max_text_length": 29,
                    "unique_texts": 2,
                    "min_human_summaries_length": 2,
                    "avg_human_summaries_length": 2.0,
                    "max_human_summaries_length": 2,
                    "unique_human_summaries": 2,
                    "min_machine_summaries_length": 2,
                    "avg_machine_summaries_length": 2.0,
                    "max_machine_summaries_length": 2,
                    "unique_machine_summaries": 2,
                    "min_relevance": [0, 1],
                    "avg_relevance": 0.5,
                    "max_relevance": [1, 0],
                },
            },
        }
    }

    metadata = TaskMetadata(
        type="Summarization",
        name="MockMultilingualSummarizationTask",
        main_score="cosine_spearman",
        **general_args,  # type: ignore
    )
    metadata.eval_langs = multilingual_eval_langs

    def load_data(self, **kwargs):
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

    @property
    def metadata_dict(self) -> dict[str, str]:
        metadata_dict = super().metadata_dict
        metadata_dict["min_score"] = 0
        metadata_dict["max_score"] = 1
        return metadata_dict


class MockRerankingTask(AbsTaskReranking):
    expected_stats = {
        "test": {
            "num_samples": 2,
            "number_of_characters": 172,
            "num_positive": 2,
            "num_negative": 2,
            "min_query_length": 23,
            "avg_query_length": 26.0,
            "max_query_length": 29,
            "unique_query": 2,
            "min_positive_length": 27,
            "avg_positive_length": 30.0,
            "max_positive_length": 33,
            "unique_positive": 2,
            "min_negative_length": 27,
            "avg_negative_length": 30.0,
            "max_negative_length": 33,
            "unique_negative": 2,
        }
    }

    metadata = TaskMetadata(
        type="Reranking",
        name="MockRerankingTask",
        main_score="map",
        **general_args,  # type: ignore
    )

    def load_data(self, **kwargs):
        query = ["This is a test sentence", "This is another test sentence"]
        positive = [
            "This is a positive sentence",
            "This is another positive sentence",
        ]
        negative = [
            "This is a negative sentence",
            "This is another negative sentence",
        ]

        self.dataset = DatasetDict(
            {
                "test": Dataset.from_dict(
                    {
                        "query": query,
                        "positive": positive,
                        "negative": negative,
                    }
                ),
            }
        )
        self.data_loaded = True


class MockMultilingualRerankingTask(AbsTaskReranking, MultilingualTask):
    expected_stats = {
        "test": {
            "num_samples": 4,
            "number_of_characters": 344,
            "num_positive": 4,
            "num_negative": 4,
            "min_query_length": 23,
            "avg_query_length": 26.0,
            "max_query_length": 29,
            "unique_query": 2,
            "min_positive_length": 27,
            "avg_positive_length": 30.0,
            "max_positive_length": 33,
            "unique_positive": 2,
            "min_negative_length": 27,
            "avg_negative_length": 30.0,
            "max_negative_length": 33,
            "unique_negative": 2,
            "hf_subset_descriptive_stats": {
                "eng": {
                    "num_samples": 2,
                    "number_of_characters": 172,
                    "num_positive": 2,
                    "num_negative": 2,
                    "min_query_length": 23,
                    "avg_query_length": 26.0,
                    "max_query_length": 29,
                    "unique_query": 2,
                    "min_positive_length": 27,
                    "avg_positive_length": 30.0,
                    "max_positive_length": 33,
                    "unique_positive": 2,
                    "min_negative_length": 27,
                    "avg_negative_length": 30.0,
                    "max_negative_length": 33,
                    "unique_negative": 2,
                },
                "fra": {
                    "num_samples": 2,
                    "number_of_characters": 172,
                    "num_positive": 2,
                    "num_negative": 2,
                    "min_query_length": 23,
                    "avg_query_length": 26.0,
                    "max_query_length": 29,
                    "unique_query": 2,
                    "min_positive_length": 27,
                    "avg_positive_length": 30.0,
                    "max_positive_length": 33,
                    "unique_positive": 2,
                    "min_negative_length": 27,
                    "avg_negative_length": 30.0,
                    "max_negative_length": 33,
                    "unique_negative": 2,
                },
            },
        }
    }

    metadata = TaskMetadata(
        type="Reranking",
        name="MockMultilingualRerankingTask",
        main_score="map",
        **general_args,  # type: ignore
    )
    metadata.eval_langs = multilingual_eval_langs

    def load_data(self, **kwargs):
        query = ["This is a test sentence", "This is another test sentence"]
        positive = [
            "This is a positive sentence",
            "This is another positive sentence",
        ]
        negative = [
            "This is a negative sentence",
            "This is another negative sentence",
        ]
        data = {
            "test": Dataset.from_dict(
                {
                    "query": query,
                    "positive": positive,
                    "negative": negative,
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


class MockRetrievalTask(AbsTaskRetrieval):
    expected_stats = {
        "test": {
            "number_of_characters": 112,
            "num_samples": 4,
            "num_queries": 2,
            "num_documents": 2,
            "min_document_length": 23,
            "average_document_length": 26.0,
            "max_document_length": 29,
            "unique_documents": 2,
            "min_query_length": 27,
            "average_query_length": 30.0,
            "max_query_length": 33,
            "unique_queries": 2,
            "min_relevant_docs_per_query": 2,
            "average_relevant_docs_per_query": 2.0,
            "max_relevant_docs_per_query": 2,
            "unique_relevant_docs": 2,
        },
        "val": {
            "number_of_characters": 112,
            "num_samples": 4,
            "num_queries": 2,
            "num_documents": 2,
            "min_document_length": 23,
            "average_document_length": 26.0,
            "max_document_length": 29,
            "unique_documents": 2,
            "min_query_length": 27,
            "average_query_length": 30.0,
            "max_query_length": 33,
            "unique_queries": 2,
            "min_relevant_docs_per_query": 2,
            "average_relevant_docs_per_query": 2.0,
            "max_relevant_docs_per_query": 2,
            "unique_relevant_docs": 2,
        },
    }

    metadata = TaskMetadata(
        type="Retrieval",
        name="MockRetrievalTask",
        main_score="ndcg_at_10",
        **dict(general_args | {"eval_splits": ["val", "test"]}),  # type: ignore
    )

    def load_data(self, **kwargs):
        self.queries = {
            "test": {
                "q1": "This is a test sentence",
                "q2": "This is another test sentence",
            },
            "val": {
                "q1": "This is a test sentence",
                "q2": "This is another test sentence",
            },
        }
        self.corpus = {
            "test": {
                "d1": "This is a positive sentence",
                "d2": "This is another positive sentence",
            },
            "val": {
                "d1": "This is a positive sentence",
                "d2": "This is another positive sentence",
            },
        }

        self.relevant_docs = {
            "test": {
                "q1": {"d1": 1, "d2": 0},
                "q2": {"d1": 0, "d2": 1},
            },
            "val": {
                "q1": {"d1": 1, "d2": 0},
                "q2": {"d1": 0, "d2": 1},
            },
        }
        self.data_loaded = True


class MockMultilingualRetrievalTask(AbsTaskRetrieval, MultilingualTask):
    expected_stats = {
        "val": {
            "number_of_characters": 224,
            "num_samples": 8,
            "num_queries": 4,
            "num_documents": 4,
            "min_document_length": 23,
            "average_document_length": 26.0,
            "max_document_length": 29,
            "unique_documents": 4,
            "min_query_length": 27,
            "average_query_length": 30.0,
            "max_query_length": 33,
            "unique_queries": 4,
            "min_relevant_docs_per_query": 2,
            "average_relevant_docs_per_query": 2.0,
            "max_relevant_docs_per_query": 2,
            "unique_relevant_docs": 4,
            "hf_subset_descriptive_stats": {
                "eng": {
                    "number_of_characters": 112,
                    "num_samples": 4,
                    "num_queries": 2,
                    "num_documents": 2,
                    "min_document_length": 23,
                    "average_document_length": 26.0,
                    "max_document_length": 29,
                    "unique_documents": 2,
                    "min_query_length": 27,
                    "average_query_length": 30.0,
                    "max_query_length": 33,
                    "unique_queries": 2,
                    "min_relevant_docs_per_query": 2,
                    "average_relevant_docs_per_query": 2.0,
                    "max_relevant_docs_per_query": 2,
                    "unique_relevant_docs": 2,
                },
                "fra": {
                    "number_of_characters": 112,
                    "num_samples": 4,
                    "num_queries": 2,
                    "num_documents": 2,
                    "min_document_length": 23,
                    "average_document_length": 26.0,
                    "max_document_length": 29,
                    "unique_documents": 2,
                    "min_query_length": 27,
                    "average_query_length": 30.0,
                    "max_query_length": 33,
                    "unique_queries": 2,
                    "min_relevant_docs_per_query": 2,
                    "average_relevant_docs_per_query": 2.0,
                    "max_relevant_docs_per_query": 2,
                    "unique_relevant_docs": 2,
                },
            },
        },
        "test": {
            "number_of_characters": 224,
            "num_samples": 8,
            "num_queries": 4,
            "num_documents": 4,
            "min_document_length": 23,
            "average_document_length": 26.0,
            "max_document_length": 29,
            "unique_documents": 4,
            "min_query_length": 27,
            "average_query_length": 30.0,
            "max_query_length": 33,
            "unique_queries": 4,
            "min_relevant_docs_per_query": 2,
            "average_relevant_docs_per_query": 2.0,
            "max_relevant_docs_per_query": 2,
            "unique_relevant_docs": 4,
            "hf_subset_descriptive_stats": {
                "eng": {
                    "number_of_characters": 112,
                    "num_samples": 4,
                    "num_queries": 2,
                    "num_documents": 2,
                    "min_document_length": 23,
                    "average_document_length": 26.0,
                    "max_document_length": 29,
                    "unique_documents": 2,
                    "min_query_length": 27,
                    "average_query_length": 30.0,
                    "max_query_length": 33,
                    "unique_queries": 2,
                    "min_relevant_docs_per_query": 2,
                    "average_relevant_docs_per_query": 2.0,
                    "max_relevant_docs_per_query": 2,
                    "unique_relevant_docs": 2,
                },
                "fra": {
                    "number_of_characters": 112,
                    "num_samples": 4,
                    "num_queries": 2,
                    "num_documents": 2,
                    "min_document_length": 23,
                    "average_document_length": 26.0,
                    "max_document_length": 29,
                    "unique_documents": 2,
                    "min_query_length": 27,
                    "average_query_length": 30.0,
                    "max_query_length": 33,
                    "unique_queries": 2,
                    "min_relevant_docs_per_query": 2,
                    "average_relevant_docs_per_query": 2.0,
                    "max_relevant_docs_per_query": 2,
                    "unique_relevant_docs": 2,
                },
            },
        },
    }

    metadata = TaskMetadata(
        type="Retrieval",
        name="MockMultilingualRetrievalTask",
        main_score="ndcg_at_10",
        **dict(general_args | {"eval_splits": ["val", "test"]}),  # type: ignore
    )
    metadata.eval_langs = multilingual_eval_langs

    def load_data(self, **kwargs):
        queries = {
            "test": {
                "q1": "This is a test sentence",
                "q2": "This is another test sentence",
            },
            "val": {
                "q1": "This is a test sentence",
                "q2": "This is another test sentence",
            },
        }
        self.queries = {"eng": queries, "fra": queries}
        corpus = {
            "test": {
                "d1": "This is a positive sentence",
                "d2": "This is another positive sentence",
            },
            "val": {
                "d1": "This is a positive sentence",
                "d2": "This is another positive sentence",
            },
        }
        self.corpus = {"eng": corpus, "fra": corpus}

        relevant_docs = {
            "test": {
                "q1": {"d1": 1, "d2": 0},
                "q2": {"d1": 0, "d2": 1},
            },
            "val": {
                "q1": {"d1": 1, "d2": 0},
                "q2": {"d1": 0, "d2": 1},
            },
        }
        self.relevant_docs = {
            "eng": relevant_docs,
            "fra": relevant_docs,
        }
        self.data_loaded = True


class MockMultilabelClassification(AbsTaskMultilabelClassification):
    expected_stats = {
        "test": {
            "num_samples": 6,
            "number_of_characters": 156,
            "number_texts_in_train": 1,
            "min_text_length": 23,
            "average_text_length": 26.0,
            "max_text_length": 29,
            "unique_texts": 2,
            "min_labels_per_text": 2,
            "average_label_per_text": 2.0,
            "max_labels_per_text": 2,
            "unique_labels": 2,
            "labels": {"0": {"count": 6}, "1": {"count": 6}},
        },
        "train": {
            "num_samples": 6,
            "number_of_characters": 159,
            "number_texts_in_train": None,
            "min_text_length": 23,
            "average_text_length": 26.5,
            "max_text_length": 30,
            "unique_texts": 2,
            "min_labels_per_text": 2,
            "average_label_per_text": 2.0,
            "max_labels_per_text": 2,
            "unique_labels": 2,
            "labels": {"0": {"count": 6}, "1": {"count": 6}},
        },
    }

    metadata = TaskMetadata(
        type="MultilabelClassification",
        name="MockMultilabelClassification",
        main_score="lrap",
        **general_args,  # type: ignore
    )

    def load_data(self, **kwargs):
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


class MockMultilingualMultilabelClassification(
    AbsTaskMultilabelClassification, MultilingualTask
):
    expected_stats = {
        "test": {
            "num_samples": 12,
            "number_of_characters": 312,
            "number_texts_in_train": 1,
            "min_text_length": 23,
            "average_text_length": 26.0,
            "max_text_length": 29,
            "unique_texts": 2,
            "min_labels_per_text": 2,
            "average_label_per_text": 2.0,
            "max_labels_per_text": 2,
            "unique_labels": 2,
            "labels": {"0": {"count": 12}, "1": {"count": 12}},
            "hf_subset_descriptive_stats": {
                "eng": {
                    "num_samples": 6,
                    "number_of_characters": 156,
                    "number_texts_in_train": 1,
                    "min_text_length": 23,
                    "average_text_length": 26.0,
                    "max_text_length": 29,
                    "unique_texts": 2,
                    "min_labels_per_text": 2,
                    "average_label_per_text": 2.0,
                    "max_labels_per_text": 2,
                    "unique_labels": 2,
                    "labels": {"0": {"count": 6}, "1": {"count": 6}},
                },
                "fra": {
                    "num_samples": 6,
                    "number_of_characters": 156,
                    "number_texts_in_train": 1,
                    "min_text_length": 23,
                    "average_text_length": 26.0,
                    "max_text_length": 29,
                    "unique_texts": 2,
                    "min_labels_per_text": 2,
                    "average_label_per_text": 2.0,
                    "max_labels_per_text": 2,
                    "unique_labels": 2,
                    "labels": {"0": {"count": 6}, "1": {"count": 6}},
                },
            },
        },
        "train": {
            "num_samples": 12,
            "number_of_characters": 318,
            "number_texts_in_train": None,
            "min_text_length": 23,
            "average_text_length": 26.5,
            "max_text_length": 30,
            "unique_texts": 2,
            "min_labels_per_text": 2,
            "average_label_per_text": 2.0,
            "max_labels_per_text": 2,
            "unique_labels": 2,
            "labels": {"0": {"count": 12}, "1": {"count": 12}},
            "hf_subset_descriptive_stats": {
                "eng": {
                    "num_samples": 6,
                    "number_of_characters": 159,
                    "number_texts_in_train": None,
                    "min_text_length": 23,
                    "average_text_length": 26.5,
                    "max_text_length": 30,
                    "unique_texts": 2,
                    "min_labels_per_text": 2,
                    "average_label_per_text": 2.0,
                    "max_labels_per_text": 2,
                    "unique_labels": 2,
                    "labels": {"0": {"count": 6}, "1": {"count": 6}},
                },
                "fra": {
                    "num_samples": 6,
                    "number_of_characters": 159,
                    "number_texts_in_train": None,
                    "min_text_length": 23,
                    "average_text_length": 26.5,
                    "max_text_length": 30,
                    "unique_texts": 2,
                    "min_labels_per_text": 2,
                    "average_label_per_text": 2.0,
                    "max_labels_per_text": 2,
                    "unique_labels": 2,
                    "labels": {"0": {"count": 6}, "1": {"count": 6}},
                },
            },
        },
    }

    metadata = TaskMetadata(
        type="MultilabelClassification",
        name="MockMultilingualMultilabelClassification",
        main_score="lrap",
        **general_args,  # type: ignore
    )
    metadata.eval_langs = multilingual_eval_langs

    def load_data(self, **kwargs):
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


class MockInstructionRetrival(AbsTaskInstructionRetrieval):
    do_length_ablation = True
    expected_stats = {
        "test": {
            "num_samples": 4,
            "num_docs": 2,
            "num_queries": 2,
            "number_of_characters": 244,
            "min_document_length": 27,
            "average_document_length": 30.0,
            "max_document_length": 33,
            "unique_docs": 2,
            "min_query_length": 23,
            "average_query_length": 26.0,
            "max_query_length": 29,
            "unique_queries": 2,
            "min_instruction_length": 26,
            "average_instruction_length": 29.0,
            "max_instruction_length": 32,
            "unique_instructions": 2,
            "min_changed_instruction_length": 34,
            "average_changed_instruction_length": 37.0,
            "max_changed_instruction_length": 40,
            "unique_changed_instructions": 2,
            "min_average_relevant_docs_per_query": 1,
            "average_relevant_docs_per_query": 1.0,
            "max_average_relevant_docs_per_query": 1,
            "min_average_top_ranked_per_query": 2,
            "average_top_ranked_per_query": 2.0,
            "max_average_top_ranked_per_query": 2,
        }
    }

    metadata = TaskMetadata(
        type="InstructionRetrieval",
        name="MockInstructionRetrival",
        main_score="p-MRR",
        **general_args,  # type: ignore
    )

    def load_data(self, **kwargs):
        self.queries = {
            "test": {
                "q1": "This is a test sentence",
                "q2": "This is another test sentence",
            }
        }
        self.corpus = {
            "test": {
                "d1": {"text": "This is a positive sentence"},
                "d2": {"text": "This is another positive sentence"},
            }
        }

        self.og_relevant_docs = {
            "test": {
                "q1": {"d1": 1, "d2": 0},
                "q2": {"d1": 0, "d2": 1},
            },
        }
        self.og_instructions = {
            "test": {
                "This is a test sentence": "This is a test instruction",
                "This is another test sentence": "This is another test instruction",
            }
        }
        self.changed_instructions = {
            "test": {
                "This is a test sentence": "This is a changed test instruction",
                "This is another test sentence": "This is changed another test instruction",
            }
        }
        self.changed_relevant_docs = {
            "test": {
                "q1": {"d1": 0, "d2": 1},
                "q2": {"d1": 1, "d2": 0},
            }
        }

        self.top_ranked = {
            "test": {
                "q1": ["d1", "d2"],
                "q2": ["d2", "d1"],
            }
        }

        self.keywords = {
            "test": {
                "This is a test sentence": "test1",
                "This is another test sentence": "test2",
            }
        }
        self.short_instructions = {
            "test": {
                "This is a test sentence": "short1",
                "This is another test sentence": "short2",
            }
        }
        self.data_loaded = True


class MockMultilingualInstructionRetrival(
    AbsTaskInstructionRetrieval, MultilingualTask
):
    do_length_ablation = True
    expected_stats = {
        "test": {
            "num_samples": 8,
            "num_docs": 4,
            "num_queries": 4,
            "number_of_characters": 488,
            "min_document_length": 27,
            "average_document_length": 30.0,
            "max_document_length": 33,
            "unique_docs": 2,
            "min_query_length": 23,
            "average_query_length": 26.0,
            "max_query_length": 29,
            "unique_queries": 2,
            "min_instruction_length": 26,
            "average_instruction_length": 29.0,
            "max_instruction_length": 32,
            "unique_instructions": 2,
            "min_changed_instruction_length": 34,
            "average_changed_instruction_length": 37.0,
            "max_changed_instruction_length": 40,
            "unique_changed_instructions": 2,
            "min_average_relevant_docs_per_query": 1,
            "average_relevant_docs_per_query": 1.0,
            "max_average_relevant_docs_per_query": 1,
            "min_average_top_ranked_per_query": 2,
            "average_top_ranked_per_query": 2.0,
            "max_average_top_ranked_per_query": 2,
            "hf_subset_descriptive_stats": {
                "eng": {
                    "num_samples": 4,
                    "num_docs": 2,
                    "num_queries": 2,
                    "number_of_characters": 244,
                    "min_document_length": 27,
                    "average_document_length": 30.0,
                    "max_document_length": 33,
                    "unique_docs": 2,
                    "min_query_length": 23,
                    "average_query_length": 26.0,
                    "max_query_length": 29,
                    "unique_queries": 2,
                    "min_instruction_length": 26,
                    "average_instruction_length": 29.0,
                    "max_instruction_length": 32,
                    "unique_instructions": 2,
                    "min_changed_instruction_length": 34,
                    "average_changed_instruction_length": 37.0,
                    "max_changed_instruction_length": 40,
                    "unique_changed_instructions": 2,
                    "min_average_relevant_docs_per_query": 1,
                    "average_relevant_docs_per_query": 1.0,
                    "max_average_relevant_docs_per_query": 1,
                    "min_average_top_ranked_per_query": 2,
                    "average_top_ranked_per_query": 2.0,
                    "max_average_top_ranked_per_query": 2,
                },
                "fra": {
                    "num_samples": 4,
                    "num_docs": 2,
                    "num_queries": 2,
                    "number_of_characters": 244,
                    "min_document_length": 27,
                    "average_document_length": 30.0,
                    "max_document_length": 33,
                    "unique_docs": 2,
                    "min_query_length": 23,
                    "average_query_length": 26.0,
                    "max_query_length": 29,
                    "unique_queries": 2,
                    "min_instruction_length": 26,
                    "average_instruction_length": 29.0,
                    "max_instruction_length": 32,
                    "unique_instructions": 2,
                    "min_changed_instruction_length": 34,
                    "average_changed_instruction_length": 37.0,
                    "max_changed_instruction_length": 40,
                    "unique_changed_instructions": 2,
                    "min_average_relevant_docs_per_query": 1,
                    "average_relevant_docs_per_query": 1.0,
                    "max_average_relevant_docs_per_query": 1,
                    "min_average_top_ranked_per_query": 2,
                    "average_top_ranked_per_query": 2.0,
                    "max_average_top_ranked_per_query": 2,
                },
            },
        }
    }

    metadata = TaskMetadata(
        type="InstructionRetrieval",
        name="MockMultilingualInstructionRetrival",
        main_score="p-MRR",
        **general_args,  # type: ignore
    )
    metadata.eval_langs = multilingual_eval_langs

    def load_data(self, **kwargs):
        queries = {
            "test": {
                "q1": "This is a test sentence",
                "q2": "This is another test sentence",
            }
        }
        self.queries = {
            "eng": queries,
            "fra": queries,
        }
        corpus = {
            "test": {
                "d1": {"text": "This is a positive sentence"},
                "d2": {"text": "This is another positive sentence"},
            }
        }
        self.corpus = {
            "eng": corpus,
            "fra": corpus,
        }

        og_relevant_docs = {
            "test": {
                "q1": {"d1": 1, "d2": 0},
                "q2": {"d1": 0, "d2": 1},
            },
        }
        self.og_relevant_docs = {
            "eng": og_relevant_docs,
            "fra": og_relevant_docs,
        }

        og_instructions = {
            "test": {
                "This is a test sentence": "This is a test instruction",
                "This is another test sentence": "This is another test instruction",
            }
        }
        self.og_instructions = {
            "eng": og_instructions,
            "fra": og_instructions,
        }
        changed_instructions = {
            "test": {
                "This is a test sentence": "This is a changed test instruction",
                "This is another test sentence": "This is changed another test instruction",
            }
        }
        self.changed_instructions = {
            "eng": changed_instructions,
            "fra": changed_instructions,
        }
        changed_relevant_docs = {
            "test": {
                "q1": {"d1": 0, "d2": 1},
                "q2": {"d1": 1, "d2": 0},
            }
        }
        self.changed_relevant_docs = {
            "eng": changed_relevant_docs,
            "fra": changed_relevant_docs,
        }

        top_ranked = {
            "test": {
                "q1": ["d1", "d2"],
                "q2": ["d2", "d1"],
            }
        }
        self.top_ranked = {
            "eng": top_ranked,
            "fra": top_ranked,
        }

        keywords = {
            "test": {
                "This is a test sentence": "test1",
                "This is another test sentence": "test2",
            }
        }
        self.keywords = {
            "eng": keywords,
            "fra": keywords,
        }
        short_instructions = {
            "test": {
                "This is a test sentence": "short1",
                "This is another test sentence": "short2",
            }
        }
        self.short_instructions = {
            "eng": short_instructions,
            "fra": short_instructions,
        }
        self.data_loaded = True


class MockMultiChoiceTask(AbsTaskAny2AnyMultiChoice):
    metadata = TaskMetadata(
        type="Any2AnyMultiChoice",
        name="MockMultiChoice",
        main_score="accuracy",
        descriptive_stats={
            "test": {
                "num_samples": 2,
                "average_question_length": 26.0,
                "average_choice_length": 30.5,
                "unique_labels": 2,
                "labels": {"1": {"count": 1}, "0": {"count": 1}},
            }
        },
        **general_args,  # type: ignore
    )
    metadata.modalities = ["image", "text"]
    metadata.category = "it2i"

    def load_data(self, **kwargs):
        images = [np.random.randint(0, 255, (100, 100, 3)) for _ in range(2)]
        images = [
            Image.fromarray(image.astype("uint8")).convert("RGBA") for image in images
        ]

        self.corpus = {
            "test": Dataset.from_dict(
                {
                    "id": ["d1", "d2"],
                    "image": [images[i] for i in range(2)],
                    "modality": ["image" for _ in range(2)],
                }
            )
        }

        self.queries = {
            "test": Dataset.from_dict(
                {
                    "id": [f"q{i}" for i in range(2)],
                    "image": [images[i] for i in range(2)],
                    "text": [
                        "This is a positive sentence",
                        "This is another positive sentence",
                    ],
                    "modality": ["image,text" for _ in range(2)],
                }
            )
        }

        self.relevant_docs = {
            "test": {
                "q0": {"d1": 1, "d2": 0},
                "q1": {"d1": 0, "d2": 1},
            },
        }
        self.data_loaded = True


class MockMultilingualMultiChoiceTask(AbsTaskAny2AnyMultiChoice, MultilingualTask):
    metadata = TaskMetadata(
        type="Any2AnyMultiChoice",
        name="MockMultilingualMultiChoice",
        main_score="accuracy",
        descriptive_stats={
            "test": {
                "num_samples": 4,
                "average_question_length": 26.0,
                "average_choice_length": 30.5,
                "unique_labels": 2,
                "labels": {"1": {"count": 2}, "0": {"count": 2}},
                "hf_subset_descriptive_stats": {
                    "eng": {
                        "num_samples": 2,
                        "average_question_length": 26.0,
                        "average_choice_length": 30.5,
                        "unique_labels": 2,
                        "labels": {"1": {"count": 1}, "0": {"count": 1}},
                    },
                    "fra": {
                        "num_samples": 2,
                        "average_question_length": 26.0,
                        "average_choice_length": 30.5,
                        "unique_labels": 2,
                        "labels": {"1": {"count": 1}, "0": {"count": 1}},
                    },
                },
            }
        },
        **general_args,  # type: ignore
    )
    metadata.eval_langs = multilingual_eval_langs
    metadata.modalities = ["image", "text"]
    metadata.category = "it2i"

    def load_data(self, **kwargs):
        images = [np.random.randint(0, 255, (100, 100, 3)) for _ in range(2)]
        images = [
            Image.fromarray(image.astype("uint8")).convert("RGBA") for image in images
        ]

        corpus = {
            "test": Dataset.from_dict(
                {
                    "id": ["d1", "d2"],
                    "image": [images[i] for i in range(2)],
                    "modality": ["image" for _ in range(2)],
                }
            )
        }
        self.corpus = {
            "eng": corpus,
            "fra": corpus,
        }

        queries = {
            "test": Dataset.from_dict(
                {
                    "id": [f"q{i}" for i in range(2)],
                    "image": [images[i] for i in range(2)],
                    "text": [
                        "This is a positive sentence",
                        "This is another positive sentence",
                    ],
                    "modality": ["image,text" for _ in range(2)],
                }
            )
        }
        self.queries = {
            "eng": queries,
            "fra": queries,
        }

        relevant_docs = {
            "test": {
                "q0": {"d1": 1, "d2": 0},
                "q1": {"d1": 0, "d2": 1},
            },
        }
        self.relevant_docs = {
            "eng": relevant_docs,
            "fra": relevant_docs,
        }

        self.data_loaded = True


class MockAny2AnyRetrievalI2TTask(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        type="Any2AnyRetrieval",
        name="MockAny2AnyRetrievalI2T",
        main_score="ndcg_at_10",
        descriptive_stats={
            "test": {
                "average_document_length": 30.0,
                "average_query_length": 26.0,
                "num_documents": 2,
                "num_queries": 2,
                "average_relevant_docs_per_query": 1.0,
            }
        },
        **general_args,  # type: ignore
    )
    metadata.modalities = ["image", "text"]
    metadata.category = "i2t"

    def load_data(self, **kwargs):
        images = [np.random.randint(0, 255, (100, 100, 3)) for _ in range(2)]
        images = [
            Image.fromarray(image.astype("uint8")).convert("RGBA") for image in images
        ]

        self.queries = {
            "test": Dataset.from_dict(
                {
                    "id": [f"q{i}" for i in range(2)],
                    "image": [images[i] for i in range(2)],
                    "modality": ["image" for _ in range(2)],
                }
            )
        }
        self.corpus = {
            "test": Dataset.from_dict(
                {
                    "id": ["d1", "d2"],
                    "text": [
                        "This is a positive sentence",
                        "This is another positive sentence",
                    ],
                    "modality": ["text" for _ in range(2)],
                }
            )
        }

        self.relevant_docs = {
            "test": {
                "q0": {"d1": 1, "d2": 0},
                "q1": {"d1": 0, "d2": 1},
            },
        }
        self.data_loaded = True


class MockAny2AnyRetrievalT2ITask(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        type="Any2AnyRetrieval",
        name="MockAny2AnyRetrievalT2I",
        main_score="ndcg_at_10",
        descriptive_stats={
            "test": {
                "average_document_length": 30.0,
                "average_query_length": 26.0,
                "num_documents": 2,
                "num_queries": 2,
                "average_relevant_docs_per_query": 1.0,
            }
        },
        **general_args,  # type: ignore
    )
    metadata.modalities = ["image", "text"]
    metadata.category = "t2i"

    def load_data(self, **kwargs):
        images = [np.random.randint(0, 255, (100, 100, 3)) for _ in range(2)]
        images = [
            Image.fromarray(image.astype("uint8")).convert("RGBA") for image in images
        ]

        self.queries = {
            "test": Dataset.from_dict(
                {
                    "id": [f"q{i}" for i in range(2)],
                    "text": [
                        "This is a positive sentence",
                        "This is another positive sentence",
                    ],
                    "modality": ["text" for _ in range(2)],
                }
            )
        }
        self.corpus = {
            "test": Dataset.from_dict(
                {
                    "id": ["d1", "d2"],
                    "image": [images[i] for i in range(2)],
                    "modality": ["image" for _ in range(2)],
                }
            )
        }

        self.relevant_docs = {
            "test": {
                "q0": {"d1": 1, "d2": 0},
                "q1": {"d1": 0, "d2": 1},
            },
        }
        self.data_loaded = True


class MockImageClassificationTask(AbsTaskImageClassification):
    metadata = TaskMetadata(
        type="ImageClassification",
        name="MockImageClassification",
        main_score="accuracy",
        descriptive_stats={
            "test": {
                "num_samples": 2,
                "average_image_size": 26.0,
                "unique_labels": 2,
                "labels": {"1": {"count": 1}, "0": {"count": 1}},
            },
            "train": {
                "num_samples": 10,
                "average_image_size": 26.0,
                "unique_labels": 2,
                "labels": {"1": {"count": 5}, "0": {"count": 5}},
            },
        },
        **general_args,  # type: ignore
    )
    metadata.modalities = ["image"]
    metadata.category = "i2i"

    def __init__(self, **kwargs):
        super().__init__(n_experiments=1, samples_per_label=5, **kwargs)

    def load_data(self, **kwargs):
        images = [np.random.randint(0, 255, (100, 100, 3)) for _ in range(2)]
        images = [
            Image.fromarray(image.astype("uint8")).convert("RGBA") for image in images
        ]
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


class MockImageClassificationKNNPTTask(AbsTaskImageClassification):
    metadata = TaskMetadata(
        type="ImageClassification",
        name="MockImageClassificationKNNPT",
        main_score="accuracy",
        descriptive_stats={
            "test": {
                "num_samples": 2,
                "average_image_size": 26.0,
                "unique_labels": 2,
                "labels": {"1": {"count": 1}, "0": {"count": 1}},
            },
            "train": {
                "num_samples": 10,
                "average_image_size": 26.0,
                "unique_labels": 2,
                "labels": {"1": {"count": 5}, "0": {"count": 5}},
            },
        },
        **general_args,  # type: ignore
    )
    metadata.modalities = ["image"]
    metadata.category = "i2i"

    def __init__(self, **kwargs):
        super().__init__(
            method="kNN-pytorch", n_experiments=1, samples_per_label=5, **kwargs
        )

    def load_data(self, **kwargs):
        images = [np.random.randint(0, 255, (100, 100, 3)) for _ in range(2)]
        images = [
            Image.fromarray(image.astype("uint8")).convert("RGBA") for image in images
        ]
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


class MockImageClassificationKNNTask(AbsTaskImageClassification):
    metadata = TaskMetadata(
        type="ImageClassification",
        name="MockImageClassificationKNN",
        main_score="accuracy",
        descriptive_stats={
            "test": {
                "num_samples": 2,
                "average_image_size": 26.0,
                "unique_labels": 2,
                "labels": {"1": {"count": 1}, "0": {"count": 1}},
            },
            "train": {
                "num_samples": 10,
                "average_image_size": 26.0,
                "unique_labels": 2,
                "labels": {"1": {"count": 5}, "0": {"count": 5}},
            },
        },
        **general_args,  # type: ignore
    )
    metadata.modalities = ["image"]
    metadata.category = "i2i"

    def __init__(self, **kwargs):
        super().__init__(method="kNN", n_experiments=1, samples_per_label=5, **kwargs)

    def load_data(self, **kwargs):
        images = [np.random.randint(0, 255, (100, 100, 3)) for _ in range(2)]
        images = [
            Image.fromarray(image.astype("uint8")).convert("RGBA") for image in images
        ]
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


class MockMultilingualImageClassificationTask(
    AbsTaskImageClassification, MultilingualTask
):
    n_experiments = 1
    samples_per_label = 5
    metadata = TaskMetadata(
        type="ImageClassification",
        name="MockMultilingualImageClassification",
        main_score="accuracy",
        descriptive_stats={
            "test": {
                "num_samples": 4,
                "average_image_size": 26.0,
                "unique_labels": 2,
                "labels": {"1": {"count": 2}, "0": {"count": 2}},
                "hf_subset_descriptive_stats": {
                    "eng": {
                        "num_samples": 2,
                        "average_image_size": 26.0,
                        "unique_labels": 2,
                        "labels": {"1": {"count": 1}, "0": {"count": 1}},
                    },
                    "fra": {
                        "num_samples": 2,
                        "average_image_size": 26.0,
                        "unique_labels": 2,
                        "labels": {"1": {"count": 1}, "0": {"count": 1}},
                    },
                },
            },
            "train": {
                "num_samples": 20,
                "average_image_size": 26.0,
                "unique_labels": 2,
                "labels": {"1": {"count": 10}, "0": {"count": 10}},
                "hf_subset_descriptive_stats": {
                    "eng": {
                        "num_samples": 10,
                        "average_image_size": 26.0,
                        "unique_labels": 2,
                        "labels": {"1": {"count": 5}, "0": {"count": 5}},
                    },
                    "fra": {
                        "num_samples": 10,
                        "average_image_size": 26.0,
                        "unique_labels": 2,
                        "labels": {"1": {"count": 5}, "0": {"count": 5}},
                    },
                },
            },
        },
        **general_args,  # type: ignore
    )
    metadata.modalities = ["image"]
    metadata.category = "i2i"
    metadata.eval_langs = multilingual_eval_langs

    def load_data(self, **kwargs):
        images = [np.random.randint(0, 255, (100, 100, 3)) for _ in range(2)]
        images = [
            Image.fromarray(image.astype("uint8")).convert("RGBA") for image in images
        ]
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


class MockImageClusteringTask(AbsTaskImageClustering):
    metadata = TaskMetadata(
        type="ImageClustering",
        name="MockImageClustering",
        main_score="nmi",
        descriptive_stats={
            "test": {
                "num_samples": 2,
                "average_image_size": 26.0,
                "unique_labels": 2,
                "labels": {"1": {"count": 1}, "0": {"count": 1}},
            }
        },
        **general_args,  # type: ignore
    )
    metadata.modalities = ["image"]

    def load_data(self, **kwargs):
        images = [np.random.randint(0, 255, (100, 100, 3)) for _ in range(2)]
        images = [
            Image.fromarray(image.astype("uint8")).convert("RGBA") for image in images
        ]
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


class MockImageMultilabelClassificationTask(AbsTaskImageMultilabelClassification):
    metadata = TaskMetadata(
        type="ImageMultilabelClassification",
        name="MockImageMultilabelClassification",
        main_score="accuracy",
        descriptive_stats={
            "test": {
                "average_image_size": 26.0,
                "average_label_per_image": 2.0,
                "num_samples": 6,
                "unique_labels": 2,
                "labels": {"0": {"count": 6}, "1": {"count": 6}},
            }
        },
        **general_args,  # type: ignore
    )
    metadata.modalities = ["image"]
    metadata.category = "i2i"
    n_experiments = 1
    samples_per_label = 3

    def load_data(self, **kwargs):
        images = [np.random.randint(0, 255, (100, 100, 3)) for _ in range(2)]
        images = [
            Image.fromarray(image.astype("uint8")).convert("RGBA") for image in images
        ]
        labels = [["0", "3"], ["1", "2"]]

        self.dataset = DatasetDict(
            {
                "test": Dataset.from_dict(
                    {
                        "image": images * 2,
                        "labels": labels * 2,
                    }
                ),
                "train": Dataset.from_dict(
                    {
                        "image": images * 5,
                        "labels": labels * 5,
                    }
                ),
            }
        )
        self.data_loaded = True


class MockMultilingualImageMultilabelClassificationTask(
    AbsTaskImageMultilabelClassification, MultilingualTask
):
    metadata = TaskMetadata(
        type="ImageMultilabelClassification",
        name="MockMultilingualImageMultilabelClassification",
        main_score="accuracy",
        descriptive_stats={
            "test": {
                "average_image_size": 26.0,
                "average_label_per_image": 2.0,
                "num_samples": 12,
                "unique_labels": 2,
                "labels": {"0": {"count": 12}, "1": {"count": 12}},
                "hf_subset_descriptive_stats": {
                    "eng": {
                        "average_image_size": 26.0,
                        "average_label_per_image": 2.0,
                        "num_samples": 6,
                        "unique_labels": 2,
                        "labels": {"0": {"count": 6}, "1": {"count": 6}},
                    },
                    "fra": {
                        "average_image_size": 26.0,
                        "average_label_per_image": 2.0,
                        "num_samples": 6,
                        "unique_labels": 2,
                        "labels": {"0": {"count": 6}, "1": {"count": 6}},
                    },
                },
            }
        },
        **general_args,  # type: ignore
    )
    metadata.modalities = ["image"]
    metadata.eval_langs = multilingual_eval_langs

    def load_data(self, **kwargs):
        images = [np.random.randint(0, 255, (100, 100, 3)) for _ in range(2)]
        images = [
            Image.fromarray(image.astype("uint8")).convert("RGBA") for image in images
        ]
        labels = [["0", "3"], ["1", "2"]]

        data = {
            "test": Dataset.from_dict(
                {
                    "image": images * 2,
                    "labels": labels * 2,
                }
            ),
            "train": Dataset.from_dict(
                {
                    "image": images * 5,
                    "labels": labels * 5,
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


class MockImageTextPairClassificationTask(AbsTaskImageTextPairClassification):
    metadata = TaskMetadata(
        type="Compositionality",
        name="MockImageTextPairClassification",
        main_score="text_acc",
        descriptive_stats={
            "test": {
                "average_image_size": 26.0,
                "average_text_length": 30.0,
                "num_samples": 2,
                "unique_labels": 2,
                "labels": {"1": {"count": 1}, "0": {"count": 1}},
            }
        },
        **general_args,  # type: ignore
    )
    metadata.modalities = ["image", "text"]
    metadata.category = "i2t"

    def load_data(self, **kwargs):
        images = [np.random.randint(0, 255, (100, 100, 3)) for _ in range(2)]
        images = [
            [Image.fromarray(image.astype("uint8")).convert("RGBA") for image in images]
        ]
        texts = [["This is a test sentence", "This is another test sentence"]]

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
    AbsTaskImageTextPairClassification, MultilingualTask
):
    metadata = TaskMetadata(
        type="Compositionality",
        name="MockMultilingualImageTextPairClassification",
        main_score="accuracy",
        descriptive_stats={
            "test": {
                "average_image_size": 26.0,
                "average_text_length": 30.0,
                "num_samples": 4,
                "unique_labels": 2,
                "labels": {"1": {"count": 2}, "0": {"count": 2}},
                "hf_subset_descriptive_stats": {
                    "eng": {
                        "average_image_size": 26.0,
                        "average_text_length": 30.0,
                        "num_samples": 2,
                        "unique_labels": 2,
                        "labels": {"1": {"count": 1}, "0": {"count": 1}},
                    },
                    "fra": {
                        "average_image_size": 26.0,
                        "average_text_length": 30.0,
                        "num_samples": 2,
                        "unique_labels": 2,
                        "labels": {"1": {"count": 1}, "0": {"count": 1}},
                    },
                },
            }
        },
        **general_args,  # type: ignore
    )
    metadata.modalities = ["image", "text"]
    metadata.category = "i2t"

    metadata.eval_langs = multilingual_eval_langs

    def load_data(self, **kwargs):
        images = [np.random.randint(0, 255, (100, 100, 3)) for _ in range(2)]
        images = [
            [Image.fromarray(image.astype("uint8")).convert("RGBA") for image in images]
        ]
        texts = [["This is a test sentence", "This is another test sentence"]]
        data = {
            "test": Dataset.from_dict(
                {
                    "image": images,
                    "caption": texts,
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


class MockVisualSTSTask(AbsTaskVisualSTS):
    metadata = TaskMetadata(
        type="VisualSTS(eng)",
        name="MockVisualSTS",
        main_score="cosine_spearman",
        descriptive_stats={
            "test": {
                "average_image_size": 26.0,
                "average_text_length": 30.0,
                "num_samples": 2,
                "average_score": 0.5,
            }
        },
        **general_args,  # type: ignore
    )
    metadata.modalities = ["image", "text"]
    metadata.category = "i2i"

    def load_data(self, **kwargs):
        images = [np.random.randint(0, 255, (100, 100, 3)) for _ in range(2)]
        images = [
            Image.fromarray(image.astype("uint8")).convert("RGBA") for image in images
        ]
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
        self.data_loaded = True

    @property
    def metadata_dict(self) -> dict[str, str]:
        metadata_dict = super().metadata_dict
        metadata_dict["min_score"] = 0
        metadata_dict["max_score"] = 5
        return metadata_dict


class MockZeroShotClassificationTask(AbsTaskZeroShotClassification):
    metadata = TaskMetadata(
        type="ZeroShotClassification",
        name="MockZeroShotClassification",
        main_score="accuracy",
        descriptive_stats={
            "test": {
                "average_text_length": 26.0,
                "num_samples": 2,
                "unique_labels": 2,
                "labels": {"1": {"count": 1}, "0": {"count": 1}},
            }
        },
        **general_args,  # type: ignore
    )
    metadata.modalities = ["image", "text"]
    metadata.category = "i2t"

    def load_data(self, **kwargs):
        images = [np.random.randint(0, 255, (100, 100, 3)) for _ in range(2)]
        images = [
            Image.fromarray(image.astype("uint8")).convert("RGBA") for image in images
        ]
        labels = ["label1", "label2"]

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

    def get_candidate_labels(self) -> list[str]:
        return ["This is a test sentence", "This is another test sentence"]
