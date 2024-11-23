"""This implements minimal viable mock tasks for testing the benchmarking framework."""

from __future__ import annotations

from datasets import Dataset, DatasetDict

from mteb.abstasks import MultilingualTask
from mteb.abstasks.AbsTaskBitextMining import AbsTaskBitextMining
from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.AbsTaskClustering import AbsTaskClustering
from mteb.abstasks.AbsTaskClusteringFast import AbsTaskClusteringFast
from mteb.abstasks.AbsTaskMultilabelClassification import (
    AbsTaskMultilabelClassification,
)
from mteb.abstasks.AbsTaskPairClassification import AbsTaskPairClassification
from mteb.abstasks.AbsTaskReranking import AbsTaskReranking
from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.AbsTaskSTS import AbsTaskSTS
from mteb.abstasks.AbsTaskSummarization import AbsTaskSummarization
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
            "number_texts_intersect_with_train": 2,
            "min_text_length": 23,
            "average_text_length": 26.0,
            "max_text_length": 29,
            "unique_text": 2,
            "unique_labels": 2,
            "labels": {"0": {"count": 1}, "1": {"count": 1}},
        },
        "train": {
            "num_samples": 2,
            "number_of_characters": 52,
            "number_texts_intersect_with_train": None,
            "min_text_length": 23,
            "average_text_length": 26.0,
            "max_text_length": 29,
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
        texts = ["This is a test sentence", "This is another test sentence"]
        labels = [0, 1]

        self.dataset = DatasetDict(
            {
                "test": Dataset.from_dict(
                    {
                        "text": texts,
                        "label": labels,
                    }
                ),
                "train": Dataset.from_dict(
                    {
                        "text": texts,
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
            "number_texts_intersect_with_train": 2,
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
                    "number_texts_intersect_with_train": 2,
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
                    "number_texts_intersect_with_train": 2,
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
            "number_of_characters": 104,
            "number_texts_intersect_with_train": None,
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
                    "number_texts_intersect_with_train": None,
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
                    "number_texts_intersect_with_train": None,
                    "min_text_length": 23,
                    "average_text_length": 26.0,
                    "max_text_length": 29,
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
        texts = ["This is a test sentence", "This is another test sentence"]
        labels = [0, 1]
        data = {
            "test": Dataset.from_dict(
                {
                    "text": texts,
                    "label": labels,
                }
            ),
            "train": Dataset.from_dict(
                {
                    "text": texts,
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
            "unique_texts": 2,
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
            "unique_texts": 2,
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
                    "unique_texts": 2,
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
                    "unique_texts": 2,
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
            "unique_pairs": 2,
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
            "unique_pairs": 2,
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
                    "unique_pairs": 2,
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
                    "unique_pairs": 2,
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
            "unique_pairs": 2,
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
            "unique_pairs": 2,
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
                    "unique_pairs": 2,
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
                    "unique_pairs": 2,
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
    expected_stats = {'test': {'num_samples': 4, 'number_of_characters': 106, 'num_documents': 2, 'min_document_length': 27, 'average_document_length': 27.0, 'max_document_length': 27, 'unique_documents': 2, 'num_queries': 2, 'min_query_length': 23, 'average_query_length': 26.0, 'max_query_length': 29, 'unique_queries': 2, 'none_queries': 0, 'num_relevant_docs': 4, 'min_relevant_docs_per_query': 2, 'average_relevant_docs_per_query': 1.0, 'max_relevant_docs_per_query': 2, 'unique_relevant_docs': 2, 'num_instructions': None, 'min_instruction_length': None, 'average_instruction_length': None, 'max_instruction_length': None, 'unique_instructions': None, 'num_top_ranked': 2, 'min_top_ranked_per_query': 2, 'average_top_ranked_per_query': 2.0, 'max_top_ranked_per_query': 2}}


    metadata = TaskMetadata(
        type="Reranking",
        name="MockRerankingTask",
        main_score="map_at_1000",
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
                "d1": "This is a positive sentence",
                "d2": "This is a negative sentence",
            }
        }

        self.relevant_docs = {
            "test": {
                "q1": {"d1": 1, "d2": 0},
                "q2": {"d1": 0, "d2": 1},
            },
        }

        self.top_ranked = {
            "test": {
                "q1": ["d1", "d2"],
                "q2": ["d2", "d1"],
            },
        }
        self.instructions = None
        self.data_loaded = True


class MockMultilingualRerankingTask(AbsTaskReranking, MultilingualTask):
    expected_stats = {'test': {'num_samples': 8, 'number_of_characters': 224, 'num_documents': 4, 'min_document_length': 27, 'average_document_length': 30.0, 'max_document_length': 33, 'unique_documents': 4, 'num_queries': 4, 'min_query_length': 23, 'average_query_length': 26.0, 'max_query_length': 29, 'unique_queries': 4, 'none_queries': 0, 'num_relevant_docs': 8, 'min_relevant_docs_per_query': 2, 'average_relevant_docs_per_query': 1.0, 'max_relevant_docs_per_query': 2, 'unique_relevant_docs': 4, 'num_instructions': None, 'min_instruction_length': None, 'average_instruction_length': None, 'max_instruction_length': None, 'unique_instructions': None, 'num_top_ranked': 4, 'min_top_ranked_per_query': 2, 'average_top_ranked_per_query': 2.0, 'max_top_ranked_per_query': 2, 'hf_subset_descriptive_stats': {'eng': {'num_samples': 4, 'number_of_characters': 112, 'num_documents': 2, 'min_document_length': 27, 'average_document_length': 30.0, 'max_document_length': 33, 'unique_documents': 2, 'num_queries': 2, 'min_query_length': 23, 'average_query_length': 26.0, 'max_query_length': 29, 'unique_queries': 2, 'none_queries': 0, 'num_relevant_docs': 4, 'min_relevant_docs_per_query': 2, 'average_relevant_docs_per_query': 1.0, 'max_relevant_docs_per_query': 2, 'unique_relevant_docs': 2, 'num_instructions': None, 'min_instruction_length': None, 'average_instruction_length': None, 'max_instruction_length': None, 'unique_instructions': None, 'num_top_ranked': 2, 'min_top_ranked_per_query': 2, 'average_top_ranked_per_query': 2.0, 'max_top_ranked_per_query': 2}, 'fra': {'num_samples': 4, 'number_of_characters': 112, 'num_documents': 2, 'min_document_length': 27, 'average_document_length': 30.0, 'max_document_length': 33, 'unique_documents': 2, 'num_queries': 2, 'min_query_length': 23, 'average_query_length': 26.0, 'max_query_length': 29, 'unique_queries': 2, 'none_queries': 0, 'num_relevant_docs': 4, 'min_relevant_docs_per_query': 2, 'average_relevant_docs_per_query': 1.0, 'max_relevant_docs_per_query': 2, 'unique_relevant_docs': 2, 'num_instructions': None, 'min_instruction_length': None, 'average_instruction_length': None, 'max_instruction_length': None, 'unique_instructions': None, 'num_top_ranked': 2, 'min_top_ranked_per_query': 2, 'average_top_ranked_per_query': 2.0, 'max_top_ranked_per_query': 2}}}}

    metadata = TaskMetadata(
        type="Reranking",
        name="MockMultilingualRerankingTask",
        main_score="map_at_10",
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
        self.queries = {"eng": queries, "fra": queries}
        corpus = {
            "test": {
                "d1": "This is a positive sentence",
                "d2": "This is another positive sentence",
            }
        }
        self.corpus = {"eng": corpus, "fra": corpus}

        relevant_docs = {
            "test": {
                "q1": {"d1": 1, "d2": 0},
                "q2": {"d1": 0, "d2": 1},
            },
        }
        self.relevant_docs = {
            "eng": relevant_docs,
            "fra": relevant_docs,
        }
        top_ranked = {
            "test": {
                "q1": ["d1", "d2"],
                "q2": ["d2", "d1"],
            },
        }
        self.top_ranked = {
            "eng": top_ranked,
            "fra": top_ranked,
        }
        self.instructions = None
        self.data_loaded = True


class MockRetrievalTask(AbsTaskRetrieval):
    expected_stats = {'test': {'num_samples': 4, 'number_of_characters': 154, 'num_documents': 2, 'min_document_length': 51, 'average_document_length': 51.0, 'max_document_length': 51, 'unique_documents': 2, 'num_queries': 2, 'min_query_length': 23, 'average_query_length': 26.0, 'max_query_length': 29, 'unique_queries': 2, 'none_queries': 0, 'num_relevant_docs': 4, 'min_relevant_docs_per_query': 2, 'average_relevant_docs_per_query': 1.0, 'max_relevant_docs_per_query': 2, 'unique_relevant_docs': 2, 'num_instructions': None, 'min_instruction_length': None, 'average_instruction_length': None, 'max_instruction_length': None, 'unique_instructions': None, 'num_top_ranked': None, 'min_top_ranked_per_query': None, 'average_top_ranked_per_query': None, 'max_top_ranked_per_query': None}}


    metadata = TaskMetadata(
        type="Retrieval",
        name="MockRetrievalTask",
        main_score="ndcg_at_10",
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
                "d1": {
                    "title": "This is a positive title",
                    "text": "This is a positive sentence",
                },
                "d2": {
                    "title": "This is a negative title",
                    "text": "This is a negative sentence",
                },
            }
        }

        self.relevant_docs = {
            "test": {
                "q1": {"d1": 1, "d2": 0},
                "q2": {"d1": 0, "d2": 1},
            },
        }
        self.top_ranked = None
        self.instructions = None
        self.data_loaded = True


class MockMultilingualRetrievalTask(AbsTaskRetrieval, MultilingualTask):
    expected_stats = {'test': {'num_samples': 8, 'number_of_characters': 224, 'num_documents': 4, 'min_document_length': 27, 'average_document_length': 30.0, 'max_document_length': 33, 'unique_documents': 4, 'num_queries': 4, 'min_query_length': 23, 'average_query_length': 26.0, 'max_query_length': 29, 'unique_queries': 4, 'none_queries': 0, 'num_relevant_docs': 8, 'min_relevant_docs_per_query': 2, 'average_relevant_docs_per_query': 1.0, 'max_relevant_docs_per_query': 2, 'unique_relevant_docs': 4, 'num_instructions': None, 'min_instruction_length': None, 'average_instruction_length': None, 'max_instruction_length': None, 'unique_instructions': None, 'num_top_ranked': None, 'min_top_ranked_per_query': None, 'average_top_ranked_per_query': None, 'max_top_ranked_per_query': None, 'hf_subset_descriptive_stats': {'eng': {'num_samples': 4, 'number_of_characters': 112, 'num_documents': 2, 'min_document_length': 27, 'average_document_length': 30.0, 'max_document_length': 33, 'unique_documents': 2, 'num_queries': 2, 'min_query_length': 23, 'average_query_length': 26.0, 'max_query_length': 29, 'unique_queries': 2, 'none_queries': 0, 'num_relevant_docs': 4, 'min_relevant_docs_per_query': 2, 'average_relevant_docs_per_query': 1.0, 'max_relevant_docs_per_query': 2, 'unique_relevant_docs': 2, 'num_instructions': None, 'min_instruction_length': None, 'average_instruction_length': None, 'max_instruction_length': None, 'unique_instructions': None, 'num_top_ranked': None, 'min_top_ranked_per_query': None, 'average_top_ranked_per_query': None, 'max_top_ranked_per_query': None}, 'fra': {'num_samples': 4, 'number_of_characters': 112, 'num_documents': 2, 'min_document_length': 27, 'average_document_length': 30.0, 'max_document_length': 33, 'unique_documents': 2, 'num_queries': 2, 'min_query_length': 23, 'average_query_length': 26.0, 'max_query_length': 29, 'unique_queries': 2, 'none_queries': 0, 'num_relevant_docs': 4, 'min_relevant_docs_per_query': 2, 'average_relevant_docs_per_query': 1.0, 'max_relevant_docs_per_query': 2, 'unique_relevant_docs': 2, 'num_instructions': None, 'min_instruction_length': None, 'average_instruction_length': None, 'max_instruction_length': None, 'unique_instructions': None, 'num_top_ranked': None, 'min_top_ranked_per_query': None, 'average_top_ranked_per_query': None, 'max_top_ranked_per_query': None}}}}

    metadata = TaskMetadata(
        type="Retrieval",
        name="MockMultilingualRetrievalTask",
        main_score="ndcg_at_10",
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
        self.queries = {"eng": queries, "fra": queries}
        corpus = {
            "test": {
                "d1": "This is a positive sentence",
                "d2": "This is another positive sentence",
            }
        }
        self.corpus = {"eng": corpus, "fra": corpus}

        relevant_docs = {
            "test": {
                "q1": {"d1": 1, "d2": 0},
                "q2": {"d1": 0, "d2": 1},
            },
        }
        self.relevant_docs = {
            "eng": relevant_docs,
            "fra": relevant_docs,
        }
        self.top_ranked = None
        self.instructions = None
        self.data_loaded = True


class MockMultilabelClassification(AbsTaskMultilabelClassification):
    expected_stats = {
        "test": {
            "num_samples": 6,
            "number_of_characters": 156,
            "number_texts_intersect_with_train": 2,
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
            "number_of_characters": 156,
            "number_texts_intersect_with_train": None,
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
    }

    metadata = TaskMetadata(
        type="MultilabelClassification",
        name="MockMultilabelClassification",
        main_score="lrap",
        **general_args,  # type: ignore
    )

    def load_data(self, **kwargs):
        texts = ["This is a test sentence", "This is another test sentence"] * 3
        labels = [[0, 1], [1, 0]] * 3

        self.dataset = DatasetDict(
            {
                "test": Dataset.from_dict(
                    {
                        "text": texts,
                        "label": labels,
                    }
                ),
                "train": Dataset.from_dict(
                    {
                        "text": texts,
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
            "number_texts_intersect_with_train": 2,
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
                    "number_texts_intersect_with_train": 2,
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
                    "number_texts_intersect_with_train": 2,
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
            "number_of_characters": 312,
            "number_texts_intersect_with_train": None,
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
                    "number_texts_intersect_with_train": None,
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
                    "number_texts_intersect_with_train": None,
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
    }

    metadata = TaskMetadata(
        type="MultilabelClassification",
        name="MockMultilingualMultilabelClassification",
        main_score="lrap",
        **general_args,  # type: ignore
    )
    metadata.eval_langs = multilingual_eval_langs

    def load_data(self, **kwargs):
        texts = ["This is a test sentence", "This is another test sentence"] * 3
        labels = [[0, 1], [1, 0]] * 3

        data = {
            "test": Dataset.from_dict(
                {
                    "text": texts,
                    "label": labels,
                }
            ),
            "train": Dataset.from_dict(
                {
                    "text": texts,
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


class MockInstructionRetrieval(AbsTaskRetrieval):
    expected_stats = {'test': {'num_samples': 4, 'number_of_characters': 112, 'num_documents': 2, 'min_document_length': 27, 'average_document_length': 30.0, 'max_document_length': 33, 'unique_documents': 2, 'num_queries': 2, 'min_query_length': 23, 'average_query_length': 26.0, 'max_query_length': 29, 'unique_queries': 2, 'none_queries': 0, 'num_relevant_docs': 4, 'min_relevant_docs_per_query': 2, 'average_relevant_docs_per_query': 1.0, 'max_relevant_docs_per_query': 2, 'unique_relevant_docs': 2, 'num_instructions': 2, 'min_instruction_length': 26, 'average_instruction_length': 58, 'max_instruction_length': 32, 'unique_instructions': 2, 'num_top_ranked': None, 'min_top_ranked_per_query': None, 'average_top_ranked_per_query': None, 'max_top_ranked_per_query': None}}


    metadata = TaskMetadata(
        type="InstructionRetrieval",
        name="MockInstructionRetrieval",
        main_score="ndcg_at_10",
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
                "d1": "This is a positive sentence",
                "d2": "This is another positive sentence",
            }
        }

        self.relevant_docs = {
            "test": {
                "q1": {"d1": 1, "d2": 0},
                "q2": {"d1": 0, "d2": 1},
            },
        }
        self.instructions = {
            "test": {
                "q1": "This is a test instruction",
                "q2": "This is another test instruction",
            }
        }
        self.top_ranked = None
        self.data_loaded = True


class MockInstructionReranking(AbsTaskReranking):
    expected_stats = {'test': {'num_samples': 4, 'number_of_characters': 112, 'num_documents': 2, 'min_document_length': 27, 'average_document_length': 30.0, 'max_document_length': 33, 'unique_documents': 2, 'num_queries': 2, 'min_query_length': 23, 'average_query_length': 26.0, 'max_query_length': 29, 'unique_queries': 2, 'none_queries': 0, 'num_relevant_docs': 4, 'min_relevant_docs_per_query': 2, 'average_relevant_docs_per_query': 1.0, 'max_relevant_docs_per_query': 2, 'unique_relevant_docs': 2, 'num_instructions': 2, 'min_instruction_length': 26, 'average_instruction_length': 58, 'max_instruction_length': 32, 'unique_instructions': 2, 'num_top_ranked': 2, 'min_top_ranked_per_query': 2, 'average_top_ranked_per_query': 2.0, 'max_top_ranked_per_query': 2}}


    metadata = TaskMetadata(
        type="InstructionReranking",
        name="MockInstructionReranking",
        main_score="ndcg_at_10",
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
                "d1": "This is a positive sentence",
                "d2": "This is another positive sentence",
            }
        }

        self.relevant_docs = {
            "test": {
                "q1": {"d1": 1, "d2": 0},
                "q2": {"d1": 0, "d2": 1},
            },
        }
        self.instructions = {
            "test": {
                "q1": "This is a test instruction",
                "q2": "This is another test instruction",
            }
        }
        self.top_ranked = {
            "test": {
                "q1": ["d1", "d2"],
                "q2": ["d2", "d1"],
            }
        }
        self.data_loaded = True


class MockMultilingualInstructionRetrieval(AbsTaskRetrieval, MultilingualTask):
    expected_stats = {'test': {'num_samples': 8, 'number_of_characters': 224, 'num_documents': 4, 'min_document_length': 27, 'average_document_length': 30.0, 'max_document_length': 33, 'unique_documents': 4, 'num_queries': 4, 'min_query_length': 23, 'average_query_length': 26.0, 'max_query_length': 29, 'unique_queries': 4, 'none_queries': 0, 'num_relevant_docs': 8, 'min_relevant_docs_per_query': 2, 'average_relevant_docs_per_query': 1.0, 'max_relevant_docs_per_query': 2, 'unique_relevant_docs': 4, 'num_instructions': 4, 'min_instruction_length': 26, 'average_instruction_length': 116, 'max_instruction_length': 32, 'unique_instructions': 4, 'num_top_ranked': None, 'min_top_ranked_per_query': None, 'average_top_ranked_per_query': None, 'max_top_ranked_per_query': None, 'hf_subset_descriptive_stats': {'eng': {'num_samples': 4, 'number_of_characters': 112, 'num_documents': 2, 'min_document_length': 27, 'average_document_length': 30.0, 'max_document_length': 33, 'unique_documents': 2, 'num_queries': 2, 'min_query_length': 23, 'average_query_length': 26.0, 'max_query_length': 29, 'unique_queries': 2, 'none_queries': 0, 'num_relevant_docs': 4, 'min_relevant_docs_per_query': 2, 'average_relevant_docs_per_query': 1.0, 'max_relevant_docs_per_query': 2, 'unique_relevant_docs': 2, 'num_instructions': 2, 'min_instruction_length': 26, 'average_instruction_length': 58, 'max_instruction_length': 32, 'unique_instructions': 2, 'num_top_ranked': None, 'min_top_ranked_per_query': None, 'average_top_ranked_per_query': None, 'max_top_ranked_per_query': None}, 'fra': {'num_samples': 4, 'number_of_characters': 112, 'num_documents': 2, 'min_document_length': 27, 'average_document_length': 30.0, 'max_document_length': 33, 'unique_documents': 2, 'num_queries': 2, 'min_query_length': 23, 'average_query_length': 26.0, 'max_query_length': 29, 'unique_queries': 2, 'none_queries': 0, 'num_relevant_docs': 4, 'min_relevant_docs_per_query': 2, 'average_relevant_docs_per_query': 1.0, 'max_relevant_docs_per_query': 2, 'unique_relevant_docs': 2, 'num_instructions': 2, 'min_instruction_length': 26, 'average_instruction_length': 58, 'max_instruction_length': 32, 'unique_instructions': 2, 'num_top_ranked': None, 'min_top_ranked_per_query': None, 'average_top_ranked_per_query': None, 'max_top_ranked_per_query': None}}}}


    metadata = TaskMetadata(
        type="InstructionRetrieval",
        name="MockMultilingualInstructionRetrieval",
        main_score="ndcg_at_10",
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
                "d1": "This is a positive sentence",
                "d2": "This is another positive sentence",
            }
        }
        self.corpus = {
            "eng": corpus,
            "fra": corpus,
        }

        relevant_docs = {
            "test": {
                "q1": {"d1": 1, "d2": 0},
                "q2": {"d1": 0, "d2": 1},
            },
        }
        self.relevant_docs = {
            "eng": relevant_docs,
            "fra": relevant_docs,
        }

        instructions = {
            "test": {
                "q1": "This is a test instruction",
                "q2": "This is another test instruction",
            }
        }
        self.instructions = {
            "eng": instructions,
            "fra": instructions,
        }
        self.top_ranked = None


class MockMultilingualInstructionReranking(AbsTaskReranking, MultilingualTask):
    expected_stats = {'test': {'num_samples': 8, 'number_of_characters': 224, 'num_documents': 4, 'min_document_length': 27, 'average_document_length': 30.0, 'max_document_length': 33, 'unique_documents': 4, 'num_queries': 4, 'min_query_length': 23, 'average_query_length': 26.0, 'max_query_length': 29, 'unique_queries': 4, 'none_queries': 0, 'num_relevant_docs': 8, 'min_relevant_docs_per_query': 2, 'average_relevant_docs_per_query': 1.0, 'max_relevant_docs_per_query': 2, 'unique_relevant_docs': 4, 'num_instructions': 4, 'min_instruction_length': 26, 'average_instruction_length': 116, 'max_instruction_length': 32, 'unique_instructions': 4, 'num_top_ranked': 4, 'min_top_ranked_per_query': 2, 'average_top_ranked_per_query': 2.0, 'max_top_ranked_per_query': 2, 'hf_subset_descriptive_stats': {'eng': {'num_samples': 4, 'number_of_characters': 112, 'num_documents': 2, 'min_document_length': 27, 'average_document_length': 30.0, 'max_document_length': 33, 'unique_documents': 2, 'num_queries': 2, 'min_query_length': 23, 'average_query_length': 26.0, 'max_query_length': 29, 'unique_queries': 2, 'none_queries': 0, 'num_relevant_docs': 4, 'min_relevant_docs_per_query': 2, 'average_relevant_docs_per_query': 1.0, 'max_relevant_docs_per_query': 2, 'unique_relevant_docs': 2, 'num_instructions': 2, 'min_instruction_length': 26, 'average_instruction_length': 58, 'max_instruction_length': 32, 'unique_instructions': 2, 'num_top_ranked': 2, 'min_top_ranked_per_query': 2, 'average_top_ranked_per_query': 2.0, 'max_top_ranked_per_query': 2}, 'fra': {'num_samples': 4, 'number_of_characters': 112, 'num_documents': 2, 'min_document_length': 27, 'average_document_length': 30.0, 'max_document_length': 33, 'unique_documents': 2, 'num_queries': 2, 'min_query_length': 23, 'average_query_length': 26.0, 'max_query_length': 29, 'unique_queries': 2, 'none_queries': 0, 'num_relevant_docs': 4, 'min_relevant_docs_per_query': 2, 'average_relevant_docs_per_query': 1.0, 'max_relevant_docs_per_query': 2, 'unique_relevant_docs': 2, 'num_instructions': 2, 'min_instruction_length': 26, 'average_instruction_length': 58, 'max_instruction_length': 32, 'unique_instructions': 2, 'num_top_ranked': 2, 'min_top_ranked_per_query': 2, 'average_top_ranked_per_query': 2.0, 'max_top_ranked_per_query': 2}}}}


    metadata = TaskMetadata(
        type="InstructionReranking",
        name="MockMultilingualInstructionReranking",
        main_score="ndcg_at_10",
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
                "d1": "This is a positive sentence",
                "d2": "This is another positive sentence",
            }
        }

        self.corpus = {
            "eng": corpus,
            "fra": corpus,
        }

        relevant_docs = {
            "test": {
                "q1": {"d1": 1, "d2": 0},
                "q2": {"d1": 0, "d2": 1},
            },
        }

        self.relevant_docs = {
            "eng": relevant_docs,
            "fra": relevant_docs,
        }

        instructions = {
            "test": {
                "q1": "This is a test instruction",
                "q2": "This is another test instruction",
            }
        }
        self.instructions = {
            "eng": instructions,
            "fra": instructions,
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
        self.data_loaded = True
