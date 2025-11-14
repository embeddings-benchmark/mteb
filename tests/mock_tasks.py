"""This implements minimal viable mock tasks for testing the benchmarking framework."""

import datasets
from datasets import Dataset, DatasetDict
from PIL import Image
from sklearn.linear_model import LogisticRegression

from mteb.abstasks.aggregate_task_metadata import AggregateTaskMetadata
from mteb.abstasks.aggregated_task import AbsTaskAggregate
from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.clustering import AbsTaskClustering
from mteb.abstasks.clustering_legacy import AbsTaskClusteringLegacy
from mteb.abstasks.image.image_text_pair_classification import (
    AbsTaskImageTextPairClassification,
)
from mteb.abstasks.multilabel_classification import (
    AbsTaskMultilabelClassification,
)
from mteb.abstasks.pair_classification import AbsTaskPairClassification
from mteb.abstasks.regression import AbsTaskRegression
from mteb.abstasks.retrieval import AbsTaskRetrieval, RetrievalSplitData
from mteb.abstasks.sts import AbsTaskSTS
from mteb.abstasks.task_metadata import TaskMetadata
from mteb.abstasks.text.bitext_mining import AbsTaskBitextMining
from mteb.abstasks.text.summarization import AbsTaskSummarization
from mteb.abstasks.zeroshot_classification import (
    AbsTaskZeroShotClassification,
)

general_args = {
    "description": "a mock task for testing",
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

multilingual_eval_langs = {
    "eng": ["eng-Latn"],
    "fra": ["fra-Latn"],
}


def base_retrieval_datasplit() -> RetrievalSplitData:
    return RetrievalSplitData(
        queries=Dataset.from_list(
            [
                {
                    "id": "q1",
                    "text": "This is a test sentence",
                },
                {
                    "id": "q2",
                    "text": "This is another test sentence",
                },
            ]
        ),
        corpus=Dataset.from_list(
            [
                {
                    "id": "d2",
                    "text": "This is a positive sentence",
                    "title": "Title of d1",
                },
                {
                    "id": "d1",
                    "text": "This is another positive sentence",
                    "title": "Title of d2",
                },
            ]
        ),
        relevant_docs={
            "q1": {"d1": 1, "d2": 0},
            "q2": {"d1": 0, "d2": 1},
        },
        top_ranked={
            "q1": ["d1", "d2"],
            "q2": ["d2", "d1"],
        },
    )


def instruction_retrieval_datasplit() -> RetrievalSplitData:
    base_ds = base_retrieval_datasplit()
    base_ds["queries"] = Dataset.from_list(
        [
            {
                "id": "q1",
                "text": "This is a test sentence",
                "instruction": "This is a test instruction",
            },
            {
                "id": "q2",
                "text": "This is another test sentence",
                "instruction": "This is another test instruction",
            },
        ]
    )
    return base_ds


class MockClassificationTask(AbsTaskClassification):
    classifier = LogisticRegression(n_jobs=1, max_iter=10)  # type: ignore

    expected_stats = {
        "test": {
            "num_samples": 2,
            "number_texts_intersect_with_train": 1,
            "text_statistics": {
                "total_text_length": 52,
                "min_text_length": 23,
                "average_text_length": 26.0,
                "max_text_length": 29,
                "unique_texts": 2,
            },
            "image_statistics": None,
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
            "number_texts_intersect_with_train": None,
            "text_statistics": {
                "total_text_length": 53,
                "min_text_length": 23,
                "average_text_length": 26.5,
                "max_text_length": 30,
                "unique_texts": 2,
            },
            "image_statistics": None,
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
        **general_args,  # type: ignore
    )

    def load_data(self) -> None:
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
    classifier = LogisticRegression(n_jobs=1, max_iter=10)

    expected_stats = {
        "test": {
            "num_samples": 4,
            "number_texts_intersect_with_train": 1,
            "text_statistics": {
                "total_text_length": 104,
                "min_text_length": 23,
                "average_text_length": 26.0,
                "max_text_length": 29,
                "unique_texts": 2,
            },
            "image_statistics": None,
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
                    "number_texts_intersect_with_train": 1,
                    "text_statistics": {
                        "total_text_length": 52,
                        "min_text_length": 23,
                        "average_text_length": 26.0,
                        "max_text_length": 29,
                        "unique_texts": 2,
                    },
                    "image_statistics": None,
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
                    "number_texts_intersect_with_train": 1,
                    "text_statistics": {
                        "total_text_length": 52,
                        "min_text_length": 23,
                        "average_text_length": 26.0,
                        "max_text_length": 29,
                        "unique_texts": 2,
                    },
                    "image_statistics": None,
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
            "number_texts_intersect_with_train": None,
            "text_statistics": {
                "total_text_length": 106,
                "min_text_length": 23,
                "average_text_length": 26.5,
                "max_text_length": 30,
                "unique_texts": 2,
            },
            "image_statistics": None,
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
                    "number_texts_intersect_with_train": None,
                    "text_statistics": {
                        "total_text_length": 53,
                        "min_text_length": 23,
                        "average_text_length": 26.5,
                        "max_text_length": 30,
                        "unique_texts": 2,
                    },
                    "image_statistics": None,
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
                    "number_texts_intersect_with_train": None,
                    "text_statistics": {
                        "total_text_length": 53,
                        "min_text_length": 23,
                        "average_text_length": 26.5,
                        "max_text_length": 30,
                        "unique_texts": 2,
                    },
                    "image_statistics": None,
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
        **general_args,  # type: ignore
    )
    metadata.eval_langs = multilingual_eval_langs

    def load_data(self) -> None:
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
        **general_args,  # type: ignore
    )

    def load_data(self) -> None:
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
        **general_args,  # type: ignore
    )
    metadata.eval_langs = multilingual_eval_langs

    def load_data(self) -> None:
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
        **general_args,  # type: ignore
    )
    metadata.eval_langs = {
        "eng_Latn-fra_Latn": ["eng-Latn", "fra-Latn"],
        "fra_Latn-eng_Latn": ["eng-Latn", "fra-Latn"],
    }

    def load_data(self) -> None:
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
        **general_args,  # type: ignore
    )

    def load_data(self) -> None:
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
        **general_args,  # type: ignore
    )
    metadata.eval_langs = multilingual_eval_langs

    def load_data(self) -> None:
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
        **general_args,  # type: ignore
    )

    def load_data(self) -> None:
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
        **general_args,  # type: ignore
    )
    metadata.eval_langs = multilingual_eval_langs

    def load_data(self) -> None:
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
            "text1_statistics": {
                "total_text_length": 52,
                "min_text_length": 23,
                "average_text_length": 26.0,
                "max_text_length": 29,
                "unique_texts": 2,
            },
            "image1_statistics": None,
            "text2_statistics": {
                "total_text_length": 61,
                "min_text_length": 24,
                "average_text_length": 30.5,
                "max_text_length": 37,
                "unique_texts": 2,
            },
            "image2_statistics": None,
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
        **general_args,  # type: ignore
    )

    def load_data(self) -> None:
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
            "number_of_characters": 226,
            "unique_pairs": 2,
            "text1_statistics": {
                "total_text_length": 104,
                "min_text_length": 23,
                "average_text_length": 26.0,
                "max_text_length": 29,
                "unique_texts": 2,
            },
            "image1_statistics": None,
            "text2_statistics": {
                "total_text_length": 122,
                "min_text_length": 24,
                "average_text_length": 30.5,
                "max_text_length": 37,
                "unique_texts": 2,
            },
            "image2_statistics": None,
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
                    "number_of_characters": 113,
                    "unique_pairs": 2,
                    "text1_statistics": {
                        "total_text_length": 52,
                        "min_text_length": 23,
                        "average_text_length": 26.0,
                        "max_text_length": 29,
                        "unique_texts": 2,
                    },
                    "image1_statistics": None,
                    "text2_statistics": {
                        "total_text_length": 61,
                        "min_text_length": 24,
                        "average_text_length": 30.5,
                        "max_text_length": 37,
                        "unique_texts": 2,
                    },
                    "image2_statistics": None,
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
                    "number_of_characters": 113,
                    "unique_pairs": 2,
                    "text1_statistics": {
                        "total_text_length": 52,
                        "min_text_length": 23,
                        "average_text_length": 26.0,
                        "max_text_length": 29,
                        "unique_texts": 2,
                    },
                    "image1_statistics": None,
                    "text2_statistics": {
                        "total_text_length": 61,
                        "min_text_length": 24,
                        "average_text_length": 30.5,
                        "max_text_length": 37,
                        "unique_texts": 2,
                    },
                    "image2_statistics": None,
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
        **general_args,  # type: ignore
    )
    metadata.eval_langs = multilingual_eval_langs

    def load_data(self) -> None:
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
        **general_args,  # type: ignore
    )
    metadata.modalities = ["image"]

    input1_column_name = "image1"
    input2_column_name = "image2"

    def load_data(self) -> None:
        images1 = [self.np_rng.integers(0, 255, (100, 100, 3)) for _ in range(2)]
        images1 = [
            Image.fromarray(image.astype("uint8")).convert("RGBA") for image in images1
        ]

        images2 = [self.np_rng.integers(0, 255, (100, 100, 3)) for _ in range(2)]
        images2 = [
            Image.fromarray(image.astype("uint8")).convert("RGBA") for image in images2
        ]

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
            "label_statistics": {"min_score": 0, "avg_score": 0.5, "max_score": 1},
        }
    }

    metadata = TaskMetadata(
        type="STS",
        name="MockSTSTask",
        main_score="cosine_spearman",
        **general_args,  # type: ignore
    )

    def load_data(self) -> None:
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
        **general_args,  # type: ignore
    )
    metadata.eval_langs = multilingual_eval_langs

    def load_data(self) -> None:
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
        **general_args,  # type: ignore
    )

    def load_data(self) -> None:
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
        **general_args,  # type: ignore
    )
    metadata.eval_langs = multilingual_eval_langs

    def load_data(self) -> None:
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


class MockRerankingTask(AbsTaskRetrieval):
    expected_stats = {
        "test": {
            "num_samples": 4,
            "number_of_characters": 136,
            "documents_text_statistics": {
                "total_text_length": 84,
                "min_text_length": 39,
                "average_text_length": 42.0,
                "max_text_length": 45,
                "unique_texts": 2,
            },
            "documents_image_statistics": None,
            "queries_text_statistics": {
                "total_text_length": 52,
                "min_text_length": 23,
                "average_text_length": 26.0,
                "max_text_length": 29,
                "unique_texts": 2,
            },
            "queries_image_statistics": None,
            "relevant_docs_statistics": {
                "num_relevant_docs": 2,
                "min_relevant_docs_per_query": 2,
                "average_relevant_docs_per_query": 1.0,
                "max_relevant_docs_per_query": 2,
                "unique_relevant_docs": 2,
            },
            "top_ranked_statistics": {
                "num_top_ranked": 4,
                "min_top_ranked_per_query": 2,
                "average_top_ranked_per_query": 2.0,
                "max_top_ranked_per_query": 2,
            },
        }
    }

    metadata = TaskMetadata(
        type="Reranking",
        name="MockRerankingTask",
        main_score="map_at_1000",
        **general_args,  # type: ignore
    )

    def load_data(self) -> None:
        base_datasplit = base_retrieval_datasplit()

        self.dataset = {"default": {"test": base_datasplit}}
        self.data_loaded = True


class MockMultilingualRerankingTask(AbsTaskRetrieval):
    expected_stats = {
        "test": {
            "num_samples": 8,
            "number_of_characters": 272,
            "documents_text_statistics": {
                "total_text_length": 168,
                "min_text_length": 39,
                "average_text_length": 42.0,
                "max_text_length": 45,
                "unique_texts": 2,
            },
            "documents_image_statistics": None,
            "queries_text_statistics": {
                "total_text_length": 104,
                "min_text_length": 23,
                "average_text_length": 26.0,
                "max_text_length": 29,
                "unique_texts": 2,
            },
            "queries_image_statistics": None,
            "relevant_docs_statistics": {
                "num_relevant_docs": 4,
                "min_relevant_docs_per_query": 2,
                "average_relevant_docs_per_query": 1.0,
                "max_relevant_docs_per_query": 2,
                "unique_relevant_docs": 4,
            },
            "top_ranked_statistics": {
                "num_top_ranked": 8,
                "min_top_ranked_per_query": 2,
                "average_top_ranked_per_query": 2.0,
                "max_top_ranked_per_query": 2,
            },
            "hf_subset_descriptive_stats": {
                "eng": {
                    "num_samples": 4,
                    "number_of_characters": 136,
                    "documents_text_statistics": {
                        "total_text_length": 84,
                        "min_text_length": 39,
                        "average_text_length": 42.0,
                        "max_text_length": 45,
                        "unique_texts": 2,
                    },
                    "documents_image_statistics": None,
                    "queries_text_statistics": {
                        "total_text_length": 52,
                        "min_text_length": 23,
                        "average_text_length": 26.0,
                        "max_text_length": 29,
                        "unique_texts": 2,
                    },
                    "queries_image_statistics": None,
                    "relevant_docs_statistics": {
                        "num_relevant_docs": 2,
                        "min_relevant_docs_per_query": 2,
                        "average_relevant_docs_per_query": 1.0,
                        "max_relevant_docs_per_query": 2,
                        "unique_relevant_docs": 2,
                    },
                    "top_ranked_statistics": {
                        "num_top_ranked": 4,
                        "min_top_ranked_per_query": 2,
                        "average_top_ranked_per_query": 2.0,
                        "max_top_ranked_per_query": 2,
                    },
                },
                "fra": {
                    "num_samples": 4,
                    "number_of_characters": 136,
                    "documents_text_statistics": {
                        "total_text_length": 84,
                        "min_text_length": 39,
                        "average_text_length": 42.0,
                        "max_text_length": 45,
                        "unique_texts": 2,
                    },
                    "documents_image_statistics": None,
                    "queries_text_statistics": {
                        "total_text_length": 52,
                        "min_text_length": 23,
                        "average_text_length": 26.0,
                        "max_text_length": 29,
                        "unique_texts": 2,
                    },
                    "queries_image_statistics": None,
                    "relevant_docs_statistics": {
                        "num_relevant_docs": 2,
                        "min_relevant_docs_per_query": 2,
                        "average_relevant_docs_per_query": 1.0,
                        "max_relevant_docs_per_query": 2,
                        "unique_relevant_docs": 2,
                    },
                    "top_ranked_statistics": {
                        "num_top_ranked": 4,
                        "min_top_ranked_per_query": 2,
                        "average_top_ranked_per_query": 2.0,
                        "max_top_ranked_per_query": 2,
                    },
                },
            },
        }
    }

    metadata = TaskMetadata(
        type="Reranking",
        name="MockMultilingualRerankingTask",
        main_score="map_at_10",
        **general_args,  # type: ignore
    )
    metadata.eval_langs = multilingual_eval_langs

    def load_data(self) -> None:
        base_datasplit = base_retrieval_datasplit()
        self.dataset = {
            "eng": {"test": base_datasplit},
            "fra": {"test": base_datasplit},
        }
        self.data_loaded = True


class MockRetrievalTask(AbsTaskRetrieval):
    _top_k = 2
    expected_stats = {
        "val": {
            "num_samples": 4,
            "number_of_characters": 136,
            "documents_text_statistics": {
                "total_text_length": 84,
                "min_text_length": 39,
                "average_text_length": 42.0,
                "max_text_length": 45,
                "unique_texts": 2,
            },
            "documents_image_statistics": None,
            "queries_text_statistics": {
                "total_text_length": 52,
                "min_text_length": 23,
                "average_text_length": 26.0,
                "max_text_length": 29,
                "unique_texts": 2,
            },
            "queries_image_statistics": None,
            "relevant_docs_statistics": {
                "num_relevant_docs": 2,
                "min_relevant_docs_per_query": 2,
                "average_relevant_docs_per_query": 1.0,
                "max_relevant_docs_per_query": 2,
                "unique_relevant_docs": 2,
            },
            "top_ranked_statistics": None,
        },
        "test": {
            "num_samples": 4,
            "number_of_characters": 136,
            "documents_text_statistics": {
                "total_text_length": 84,
                "min_text_length": 39,
                "average_text_length": 42.0,
                "max_text_length": 45,
                "unique_texts": 2,
            },
            "documents_image_statistics": None,
            "queries_text_statistics": {
                "total_text_length": 52,
                "min_text_length": 23,
                "average_text_length": 26.0,
                "max_text_length": 29,
                "unique_texts": 2,
            },
            "queries_image_statistics": None,
            "relevant_docs_statistics": {
                "num_relevant_docs": 2,
                "min_relevant_docs_per_query": 2,
                "average_relevant_docs_per_query": 1.0,
                "max_relevant_docs_per_query": 2,
                "unique_relevant_docs": 2,
            },
            "top_ranked_statistics": None,
        },
    }

    metadata = TaskMetadata(
        type="Retrieval",
        name="MockRetrievalTask",
        main_score="ndcg_at_10",
        **dict(general_args | {"eval_splits": ["val", "test"]}),  # type: ignore
    )

    def load_data(self) -> None:
        base_datasplit = base_retrieval_datasplit()

        base_datasplit["top_ranked"] = None
        self.dataset = {"default": {"test": base_datasplit, "val": base_datasplit}}
        self.data_loaded = True


class MockRetrievalDialogTask(AbsTaskRetrieval):
    _top_k = 1
    expected_stats = {
        "val": {
            "num_samples": 4,
            "number_of_characters": 201,
            "documents_text_statistics": {
                "total_text_length": 84,
                "min_text_length": 39,
                "average_text_length": 42.0,
                "max_text_length": 45,
                "unique_texts": 2,
            },
            "documents_image_statistics": None,
            "queries_text_statistics": {
                "total_text_length": 117,
                "min_text_length": 37,
                "average_text_length": 58.5,
                "max_text_length": 80,
                "unique_texts": 2,
            },
            "queries_image_statistics": None,
            "relevant_docs_statistics": {
                "num_relevant_docs": 2,
                "min_relevant_docs_per_query": 2,
                "average_relevant_docs_per_query": 1.0,
                "max_relevant_docs_per_query": 2,
                "unique_relevant_docs": 2,
            },
            "top_ranked_statistics": None,
        },
        "test": {
            "num_samples": 4,
            "number_of_characters": 201,
            "documents_text_statistics": {
                "total_text_length": 84,
                "min_text_length": 39,
                "average_text_length": 42.0,
                "max_text_length": 45,
                "unique_texts": 2,
            },
            "documents_image_statistics": None,
            "queries_text_statistics": {
                "total_text_length": 117,
                "min_text_length": 37,
                "average_text_length": 58.5,
                "max_text_length": 80,
                "unique_texts": 2,
            },
            "queries_image_statistics": None,
            "relevant_docs_statistics": {
                "num_relevant_docs": 2,
                "min_relevant_docs_per_query": 2,
                "average_relevant_docs_per_query": 1.0,
                "max_relevant_docs_per_query": 2,
                "unique_relevant_docs": 2,
            },
            "top_ranked_statistics": None,
        },
    }

    metadata = TaskMetadata(
        type="Retrieval",
        name="MockRetrievalDialogTask",
        main_score="ndcg_at_10",
        **dict(general_args | {"eval_splits": ["val", "test"]}),  # type: ignore
    )

    def load_data(self) -> None:
        base_datasplit = base_retrieval_datasplit()

        base_datasplit["top_ranked"] = None
        base_datasplit["queries"] = Dataset.from_dict(
            {
                "id": ["q1", "q2"],
                "text": [
                    # dialogs with different lengths to test DatasetLoader with items with different item size
                    [
                        {"role": "user", "content": "What is the weather like today?"},
                    ],
                    [
                        {"role": "user", "content": "What is the capital of France?"},
                        {
                            "role": "assistant",
                            "content": "The capital of France is Paris.",
                        },
                    ],
                ],
            }
        )
        self.dataset = {"default": {"test": base_datasplit, "val": base_datasplit}}
        self.data_loaded = True


class MockMultilingualRetrievalTask(AbsTaskRetrieval):
    expected_stats = {
        "val": {
            "num_samples": 8,
            "number_of_characters": 272,
            "documents_text_statistics": {
                "total_text_length": 168,
                "min_text_length": 39,
                "average_text_length": 42.0,
                "max_text_length": 45,
                "unique_texts": 2,
            },
            "documents_image_statistics": None,
            "queries_text_statistics": {
                "total_text_length": 104,
                "min_text_length": 23,
                "average_text_length": 26.0,
                "max_text_length": 29,
                "unique_texts": 2,
            },
            "queries_image_statistics": None,
            "relevant_docs_statistics": {
                "num_relevant_docs": 4,
                "min_relevant_docs_per_query": 2,
                "average_relevant_docs_per_query": 1.0,
                "max_relevant_docs_per_query": 2,
                "unique_relevant_docs": 4,
            },
            "top_ranked_statistics": None,
            "hf_subset_descriptive_stats": {
                "eng": {
                    "num_samples": 4,
                    "number_of_characters": 136,
                    "documents_text_statistics": {
                        "total_text_length": 84,
                        "min_text_length": 39,
                        "average_text_length": 42.0,
                        "max_text_length": 45,
                        "unique_texts": 2,
                    },
                    "documents_image_statistics": None,
                    "queries_text_statistics": {
                        "total_text_length": 52,
                        "min_text_length": 23,
                        "average_text_length": 26.0,
                        "max_text_length": 29,
                        "unique_texts": 2,
                    },
                    "queries_image_statistics": None,
                    "relevant_docs_statistics": {
                        "num_relevant_docs": 2,
                        "min_relevant_docs_per_query": 2,
                        "average_relevant_docs_per_query": 1.0,
                        "max_relevant_docs_per_query": 2,
                        "unique_relevant_docs": 2,
                    },
                    "top_ranked_statistics": None,
                },
                "fra": {
                    "num_samples": 4,
                    "number_of_characters": 136,
                    "documents_text_statistics": {
                        "total_text_length": 84,
                        "min_text_length": 39,
                        "average_text_length": 42.0,
                        "max_text_length": 45,
                        "unique_texts": 2,
                    },
                    "documents_image_statistics": None,
                    "queries_text_statistics": {
                        "total_text_length": 52,
                        "min_text_length": 23,
                        "average_text_length": 26.0,
                        "max_text_length": 29,
                        "unique_texts": 2,
                    },
                    "queries_image_statistics": None,
                    "relevant_docs_statistics": {
                        "num_relevant_docs": 2,
                        "min_relevant_docs_per_query": 2,
                        "average_relevant_docs_per_query": 1.0,
                        "max_relevant_docs_per_query": 2,
                        "unique_relevant_docs": 2,
                    },
                    "top_ranked_statistics": None,
                },
            },
        },
        "test": {
            "num_samples": 8,
            "number_of_characters": 272,
            "documents_text_statistics": {
                "total_text_length": 168,
                "min_text_length": 39,
                "average_text_length": 42.0,
                "max_text_length": 45,
                "unique_texts": 2,
            },
            "documents_image_statistics": None,
            "queries_text_statistics": {
                "total_text_length": 104,
                "min_text_length": 23,
                "average_text_length": 26.0,
                "max_text_length": 29,
                "unique_texts": 2,
            },
            "queries_image_statistics": None,
            "relevant_docs_statistics": {
                "num_relevant_docs": 4,
                "min_relevant_docs_per_query": 2,
                "average_relevant_docs_per_query": 1.0,
                "max_relevant_docs_per_query": 2,
                "unique_relevant_docs": 4,
            },
            "top_ranked_statistics": None,
            "hf_subset_descriptive_stats": {
                "eng": {
                    "num_samples": 4,
                    "number_of_characters": 136,
                    "documents_text_statistics": {
                        "total_text_length": 84,
                        "min_text_length": 39,
                        "average_text_length": 42.0,
                        "max_text_length": 45,
                        "unique_texts": 2,
                    },
                    "documents_image_statistics": None,
                    "queries_text_statistics": {
                        "total_text_length": 52,
                        "min_text_length": 23,
                        "average_text_length": 26.0,
                        "max_text_length": 29,
                        "unique_texts": 2,
                    },
                    "queries_image_statistics": None,
                    "relevant_docs_statistics": {
                        "num_relevant_docs": 2,
                        "min_relevant_docs_per_query": 2,
                        "average_relevant_docs_per_query": 1.0,
                        "max_relevant_docs_per_query": 2,
                        "unique_relevant_docs": 2,
                    },
                    "top_ranked_statistics": None,
                },
                "fra": {
                    "num_samples": 4,
                    "number_of_characters": 136,
                    "documents_text_statistics": {
                        "total_text_length": 84,
                        "min_text_length": 39,
                        "average_text_length": 42.0,
                        "max_text_length": 45,
                        "unique_texts": 2,
                    },
                    "documents_image_statistics": None,
                    "queries_text_statistics": {
                        "total_text_length": 52,
                        "min_text_length": 23,
                        "average_text_length": 26.0,
                        "max_text_length": 29,
                        "unique_texts": 2,
                    },
                    "queries_image_statistics": None,
                    "relevant_docs_statistics": {
                        "num_relevant_docs": 2,
                        "min_relevant_docs_per_query": 2,
                        "average_relevant_docs_per_query": 1.0,
                        "max_relevant_docs_per_query": 2,
                        "unique_relevant_docs": 2,
                    },
                    "top_ranked_statistics": None,
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

    def load_data(self) -> None:
        base_datasplit = base_retrieval_datasplit()

        base_datasplit["top_ranked"] = None
        self.dataset = {
            "eng": {"test": base_datasplit, "val": base_datasplit},
            "fra": {"test": base_datasplit, "val": base_datasplit},
        }
        self.data_loaded = True


class MockMultilabelClassification(AbsTaskMultilabelClassification):
    expected_stats = {
        "test": {
            "num_samples": 6,
            "number_texts_intersect_with_train": 1,
            "text_statistics": {
                "total_text_length": 156,
                "min_text_length": 23,
                "average_text_length": 26.0,
                "max_text_length": 29,
                "unique_texts": 2,
            },
            "image_statistics": None,
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
            "number_texts_intersect_with_train": None,
            "text_statistics": {
                "total_text_length": 159,
                "min_text_length": 23,
                "average_text_length": 26.5,
                "max_text_length": 30,
                "unique_texts": 2,
            },
            "image_statistics": None,
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
        **general_args,  # type: ignore
    )

    def load_data(self) -> None:
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
            "number_texts_intersect_with_train": 1,
            "text_statistics": {
                "total_text_length": 312,
                "min_text_length": 23,
                "average_text_length": 26.0,
                "max_text_length": 29,
                "unique_texts": 2,
            },
            "image_statistics": None,
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
                    "number_texts_intersect_with_train": 1,
                    "text_statistics": {
                        "total_text_length": 156,
                        "min_text_length": 23,
                        "average_text_length": 26.0,
                        "max_text_length": 29,
                        "unique_texts": 2,
                    },
                    "image_statistics": None,
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
                    "number_texts_intersect_with_train": 1,
                    "text_statistics": {
                        "total_text_length": 156,
                        "min_text_length": 23,
                        "average_text_length": 26.0,
                        "max_text_length": 29,
                        "unique_texts": 2,
                    },
                    "image_statistics": None,
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
            "number_texts_intersect_with_train": None,
            "text_statistics": {
                "total_text_length": 318,
                "min_text_length": 23,
                "average_text_length": 26.5,
                "max_text_length": 30,
                "unique_texts": 2,
            },
            "image_statistics": None,
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
                    "number_texts_intersect_with_train": None,
                    "text_statistics": {
                        "total_text_length": 159,
                        "min_text_length": 23,
                        "average_text_length": 26.5,
                        "max_text_length": 30,
                        "unique_texts": 2,
                    },
                    "image_statistics": None,
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
                    "number_texts_intersect_with_train": None,
                    "text_statistics": {
                        "total_text_length": 159,
                        "min_text_length": 23,
                        "average_text_length": 26.5,
                        "max_text_length": 30,
                        "unique_texts": 2,
                    },
                    "image_statistics": None,
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
        **general_args,  # type: ignore
    )
    metadata.eval_langs = multilingual_eval_langs

    def load_data(self) -> None:
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


class MockInstructionRetrieval(AbsTaskRetrieval):
    expected_stats = {
        "test": {
            "num_samples": 4,
            "number_of_characters": 196,
            "documents_text_statistics": {
                "total_text_length": 84,
                "min_text_length": 39,
                "average_text_length": 42.0,
                "max_text_length": 45,
                "unique_texts": 2,
            },
            "documents_image_statistics": None,
            "queries_text_statistics": {
                "total_text_length": 112,
                "min_text_length": 50,
                "average_text_length": 56.0,
                "max_text_length": 62,
                "unique_texts": 2,
            },
            "queries_image_statistics": None,
            "relevant_docs_statistics": {
                "num_relevant_docs": 2,
                "min_relevant_docs_per_query": 2,
                "average_relevant_docs_per_query": 1.0,
                "max_relevant_docs_per_query": 2,
                "unique_relevant_docs": 2,
            },
            "top_ranked_statistics": None,
        }
    }

    metadata = TaskMetadata(
        type="InstructionRetrieval",
        name="MockInstructionRetrieval",
        main_score="ndcg_at_10",
        **general_args,  # type: ignore
    )

    def load_data(self) -> None:
        base_datasplit = instruction_retrieval_datasplit()
        base_datasplit["top_ranked"] = None

        self.dataset = {"default": {"test": base_datasplit}}
        self.data_loaded = True


class MockInstructionReranking(AbsTaskRetrieval):
    expected_stats = {
        "test": {
            "num_samples": 4,
            "number_of_characters": 196,
            "documents_text_statistics": {
                "total_text_length": 84,
                "min_text_length": 39,
                "average_text_length": 42.0,
                "max_text_length": 45,
                "unique_texts": 2,
            },
            "documents_image_statistics": None,
            "queries_text_statistics": {
                "total_text_length": 112,
                "min_text_length": 50,
                "average_text_length": 56.0,
                "max_text_length": 62,
                "unique_texts": 2,
            },
            "queries_image_statistics": None,
            "relevant_docs_statistics": {
                "num_relevant_docs": 2,
                "min_relevant_docs_per_query": 2,
                "average_relevant_docs_per_query": 1.0,
                "max_relevant_docs_per_query": 2,
                "unique_relevant_docs": 2,
            },
            "top_ranked_statistics": {
                "num_top_ranked": 4,
                "min_top_ranked_per_query": 2,
                "average_top_ranked_per_query": 2.0,
                "max_top_ranked_per_query": 2,
            },
        }
    }

    metadata = TaskMetadata(
        type="InstructionReranking",
        name="MockInstructionReranking",
        main_score="ndcg_at_10",
        **general_args,  # type: ignore
    )

    def load_data(self) -> None:
        base_datasplit = instruction_retrieval_datasplit()
        self.dataset = {"default": {"test": base_datasplit}}
        self.data_loaded = True


class MockMultilingualInstructionRetrieval(AbsTaskRetrieval):
    expected_stats = {
        "test": {
            "num_samples": 8,
            "number_of_characters": 392,
            "documents_text_statistics": {
                "total_text_length": 168,
                "min_text_length": 39,
                "average_text_length": 42.0,
                "max_text_length": 45,
                "unique_texts": 2,
            },
            "documents_image_statistics": None,
            "queries_text_statistics": {
                "total_text_length": 224,
                "min_text_length": 50,
                "average_text_length": 56.0,
                "max_text_length": 62,
                "unique_texts": 2,
            },
            "queries_image_statistics": None,
            "relevant_docs_statistics": {
                "num_relevant_docs": 4,
                "min_relevant_docs_per_query": 2,
                "average_relevant_docs_per_query": 1.0,
                "max_relevant_docs_per_query": 2,
                "unique_relevant_docs": 4,
            },
            "top_ranked_statistics": None,
            "hf_subset_descriptive_stats": {
                "eng": {
                    "num_samples": 4,
                    "number_of_characters": 196,
                    "documents_text_statistics": {
                        "total_text_length": 84,
                        "min_text_length": 39,
                        "average_text_length": 42.0,
                        "max_text_length": 45,
                        "unique_texts": 2,
                    },
                    "documents_image_statistics": None,
                    "queries_text_statistics": {
                        "total_text_length": 112,
                        "min_text_length": 50,
                        "average_text_length": 56.0,
                        "max_text_length": 62,
                        "unique_texts": 2,
                    },
                    "queries_image_statistics": None,
                    "relevant_docs_statistics": {
                        "num_relevant_docs": 2,
                        "min_relevant_docs_per_query": 2,
                        "average_relevant_docs_per_query": 1.0,
                        "max_relevant_docs_per_query": 2,
                        "unique_relevant_docs": 2,
                    },
                    "top_ranked_statistics": None,
                },
                "fra": {
                    "num_samples": 4,
                    "number_of_characters": 196,
                    "documents_text_statistics": {
                        "total_text_length": 84,
                        "min_text_length": 39,
                        "average_text_length": 42.0,
                        "max_text_length": 45,
                        "unique_texts": 2,
                    },
                    "documents_image_statistics": None,
                    "queries_text_statistics": {
                        "total_text_length": 112,
                        "min_text_length": 50,
                        "average_text_length": 56.0,
                        "max_text_length": 62,
                        "unique_texts": 2,
                    },
                    "queries_image_statistics": None,
                    "relevant_docs_statistics": {
                        "num_relevant_docs": 2,
                        "min_relevant_docs_per_query": 2,
                        "average_relevant_docs_per_query": 1.0,
                        "max_relevant_docs_per_query": 2,
                        "unique_relevant_docs": 2,
                    },
                    "top_ranked_statistics": None,
                },
            },
        }
    }

    metadata = TaskMetadata(
        type="InstructionRetrieval",
        name="MockMultilingualInstructionRetrieval",
        main_score="ndcg_at_10",
        **general_args,  # type: ignore
    )
    metadata.eval_langs = multilingual_eval_langs

    def load_data(self) -> None:
        base_datasplit = instruction_retrieval_datasplit()
        base_datasplit["top_ranked"] = None
        self.dataset = {
            "eng": {"test": base_datasplit},
            "fra": {"test": base_datasplit},
        }
        self.data_loaded = True


class MockMultilingualInstructionReranking(AbsTaskRetrieval):
    expected_stats = {
        "test": {
            "num_samples": 8,
            "number_of_characters": 392,
            "documents_text_statistics": {
                "total_text_length": 168,
                "min_text_length": 39,
                "average_text_length": 42.0,
                "max_text_length": 45,
                "unique_texts": 2,
            },
            "documents_image_statistics": None,
            "queries_text_statistics": {
                "total_text_length": 224,
                "min_text_length": 50,
                "average_text_length": 56.0,
                "max_text_length": 62,
                "unique_texts": 2,
            },
            "queries_image_statistics": None,
            "relevant_docs_statistics": {
                "num_relevant_docs": 4,
                "min_relevant_docs_per_query": 2,
                "average_relevant_docs_per_query": 1.0,
                "max_relevant_docs_per_query": 2,
                "unique_relevant_docs": 4,
            },
            "top_ranked_statistics": {
                "num_top_ranked": 8,
                "min_top_ranked_per_query": 2,
                "average_top_ranked_per_query": 2.0,
                "max_top_ranked_per_query": 2,
            },
            "hf_subset_descriptive_stats": {
                "eng": {
                    "num_samples": 4,
                    "number_of_characters": 196,
                    "documents_text_statistics": {
                        "total_text_length": 84,
                        "min_text_length": 39,
                        "average_text_length": 42.0,
                        "max_text_length": 45,
                        "unique_texts": 2,
                    },
                    "documents_image_statistics": None,
                    "queries_text_statistics": {
                        "total_text_length": 112,
                        "min_text_length": 50,
                        "average_text_length": 56.0,
                        "max_text_length": 62,
                        "unique_texts": 2,
                    },
                    "queries_image_statistics": None,
                    "relevant_docs_statistics": {
                        "num_relevant_docs": 2,
                        "min_relevant_docs_per_query": 2,
                        "average_relevant_docs_per_query": 1.0,
                        "max_relevant_docs_per_query": 2,
                        "unique_relevant_docs": 2,
                    },
                    "top_ranked_statistics": {
                        "num_top_ranked": 4,
                        "min_top_ranked_per_query": 2,
                        "average_top_ranked_per_query": 2.0,
                        "max_top_ranked_per_query": 2,
                    },
                },
                "fra": {
                    "num_samples": 4,
                    "number_of_characters": 196,
                    "documents_text_statistics": {
                        "total_text_length": 84,
                        "min_text_length": 39,
                        "average_text_length": 42.0,
                        "max_text_length": 45,
                        "unique_texts": 2,
                    },
                    "documents_image_statistics": None,
                    "queries_text_statistics": {
                        "total_text_length": 112,
                        "min_text_length": 50,
                        "average_text_length": 56.0,
                        "max_text_length": 62,
                        "unique_texts": 2,
                    },
                    "queries_image_statistics": None,
                    "relevant_docs_statistics": {
                        "num_relevant_docs": 2,
                        "min_relevant_docs_per_query": 2,
                        "average_relevant_docs_per_query": 1.0,
                        "max_relevant_docs_per_query": 2,
                        "unique_relevant_docs": 2,
                    },
                    "top_ranked_statistics": {
                        "num_top_ranked": 4,
                        "min_top_ranked_per_query": 2,
                        "average_top_ranked_per_query": 2.0,
                        "max_top_ranked_per_query": 2,
                    },
                },
            },
        }
    }

    metadata = TaskMetadata(
        type="InstructionReranking",
        name="MockMultilingualInstructionReranking",
        main_score="ndcg_at_10",
        **general_args,  # type: ignore
    )
    metadata.eval_langs = multilingual_eval_langs

    def load_data(self) -> None:
        base_datasplit = instruction_retrieval_datasplit()
        self.dataset = {
            "eng": {"test": base_datasplit, "val": base_datasplit},
            "fra": {"test": base_datasplit, "val": base_datasplit},
        }
        self.data_loaded = True


class MockAggregatedTask(AbsTaskAggregate):
    metadata = AggregateTaskMetadata(
        type="InstructionReranking",
        name="MockMultilingualInstructionReranking",
        main_score="ndcg_at_10",
        tasks=[
            MockRetrievalTask(),
            MockRerankingTask(),
        ],
        **general_args,  # type: ignore
    )


class MockMultiChoiceTask(AbsTaskRetrieval):
    expected_stats = {
        "test": {
            "num_samples": 4,
            "number_of_characters": 60,
            "documents_text_statistics": None,
            "documents_image_statistics": {
                "min_image_width": 100,
                "average_image_width": 100.0,
                "max_image_width": 100,
                "min_image_height": 100,
                "average_image_height": 100.0,
                "max_image_height": 100,
                "unique_images": 2,
            },
            "queries_text_statistics": {
                "total_text_length": 60,
                "min_text_length": 27,
                "average_text_length": 30.0,
                "max_text_length": 33,
                "unique_texts": 2,
            },
            "queries_image_statistics": {
                "min_image_width": 100,
                "average_image_width": 100.0,
                "max_image_width": 100,
                "min_image_height": 100,
                "average_image_height": 100.0,
                "max_image_height": 100,
                "unique_images": 2,
            },
            "relevant_docs_statistics": {
                "num_relevant_docs": 2,
                "min_relevant_docs_per_query": 2,
                "average_relevant_docs_per_query": 1.0,
                "max_relevant_docs_per_query": 2,
                "unique_relevant_docs": 2,
            },
            "top_ranked_statistics": {
                "num_top_ranked": 4,
                "min_top_ranked_per_query": 2,
                "average_top_ranked_per_query": 2.0,
                "max_top_ranked_per_query": 2,
            },
        }
    }

    metadata = TaskMetadata(
        type="Any2AnyMultiChoice",
        name="MockMultiChoice",
        main_score="accuracy",
        **general_args,  # type: ignore
    )
    metadata.modalities = ["image", "text"]
    metadata.category = "it2i"

    def load_data(self) -> None:
        images = [self.np_rng.integers(0, 255, (100, 100, 3)) for _ in range(2)]
        images = [
            Image.fromarray(image.astype("uint8")).convert("RGBA") for image in images
        ]
        retrieval_split_data = RetrievalSplitData(
            queries=Dataset.from_dict(
                {
                    "id": [f"q{i}" for i in range(2)],
                    "image": [images[i] for i in range(2)],
                    "text": [
                        "This is a positive sentence",
                        "This is another positive sentence",
                    ],
                    "modality": ["image,text" for _ in range(2)],
                }
            ),
            corpus=Dataset.from_dict(
                {
                    "id": ["d1", "d2"],
                    "image": [images[i] for i in range(2)],
                    "modality": ["image" for _ in range(2)],
                }
            ),
            relevant_docs={
                "q0": {"d1": 1, "d2": 0},
                "q1": {"d1": 0, "d2": 1},
            },
            top_ranked={
                "q0": ["d1", "d2"],
                "q1": ["d2", "d1"],
            },
        )
        self.dataset = {"default": {"test": retrieval_split_data}}
        self.data_loaded = True


class MockMultilingualMultiChoiceTask(AbsTaskRetrieval):
    expected_stats = {
        "test": {
            "num_samples": 8,
            "number_of_characters": 120,
            "documents_text_statistics": None,
            "documents_image_statistics": {
                "min_image_width": 100,
                "average_image_width": 100.0,
                "max_image_width": 100,
                "min_image_height": 100,
                "average_image_height": 100.0,
                "max_image_height": 100,
                "unique_images": 2,
            },
            "queries_text_statistics": {
                "total_text_length": 120,
                "min_text_length": 27,
                "average_text_length": 30.0,
                "max_text_length": 33,
                "unique_texts": 2,
            },
            "queries_image_statistics": {
                "min_image_width": 100,
                "average_image_width": 100.0,
                "max_image_width": 100,
                "min_image_height": 100,
                "average_image_height": 100.0,
                "max_image_height": 100,
                "unique_images": 2,
            },
            "relevant_docs_statistics": {
                "num_relevant_docs": 4,
                "min_relevant_docs_per_query": 2,
                "average_relevant_docs_per_query": 1.0,
                "max_relevant_docs_per_query": 2,
                "unique_relevant_docs": 4,
            },
            "top_ranked_statistics": {
                "num_top_ranked": 8,
                "min_top_ranked_per_query": 2,
                "average_top_ranked_per_query": 2.0,
                "max_top_ranked_per_query": 2,
            },
            "hf_subset_descriptive_stats": {
                "eng": {
                    "num_samples": 4,
                    "number_of_characters": 60,
                    "documents_text_statistics": None,
                    "documents_image_statistics": {
                        "min_image_width": 100,
                        "average_image_width": 100.0,
                        "max_image_width": 100,
                        "min_image_height": 100,
                        "average_image_height": 100.0,
                        "max_image_height": 100,
                        "unique_images": 2,
                    },
                    "queries_text_statistics": {
                        "total_text_length": 60,
                        "min_text_length": 27,
                        "average_text_length": 30.0,
                        "max_text_length": 33,
                        "unique_texts": 2,
                    },
                    "queries_image_statistics": {
                        "min_image_width": 100,
                        "average_image_width": 100.0,
                        "max_image_width": 100,
                        "min_image_height": 100,
                        "average_image_height": 100.0,
                        "max_image_height": 100,
                        "unique_images": 2,
                    },
                    "relevant_docs_statistics": {
                        "num_relevant_docs": 2,
                        "min_relevant_docs_per_query": 2,
                        "average_relevant_docs_per_query": 1.0,
                        "max_relevant_docs_per_query": 2,
                        "unique_relevant_docs": 2,
                    },
                    "top_ranked_statistics": {
                        "num_top_ranked": 4,
                        "min_top_ranked_per_query": 2,
                        "average_top_ranked_per_query": 2.0,
                        "max_top_ranked_per_query": 2,
                    },
                },
                "fra": {
                    "num_samples": 4,
                    "number_of_characters": 60,
                    "documents_text_statistics": None,
                    "documents_image_statistics": {
                        "min_image_width": 100,
                        "average_image_width": 100.0,
                        "max_image_width": 100,
                        "min_image_height": 100,
                        "average_image_height": 100.0,
                        "max_image_height": 100,
                        "unique_images": 2,
                    },
                    "queries_text_statistics": {
                        "total_text_length": 60,
                        "min_text_length": 27,
                        "average_text_length": 30.0,
                        "max_text_length": 33,
                        "unique_texts": 2,
                    },
                    "queries_image_statistics": {
                        "min_image_width": 100,
                        "average_image_width": 100.0,
                        "max_image_width": 100,
                        "min_image_height": 100,
                        "average_image_height": 100.0,
                        "max_image_height": 100,
                        "unique_images": 2,
                    },
                    "relevant_docs_statistics": {
                        "num_relevant_docs": 2,
                        "min_relevant_docs_per_query": 2,
                        "average_relevant_docs_per_query": 1.0,
                        "max_relevant_docs_per_query": 2,
                        "unique_relevant_docs": 2,
                    },
                    "top_ranked_statistics": {
                        "num_top_ranked": 4,
                        "min_top_ranked_per_query": 2,
                        "average_top_ranked_per_query": 2.0,
                        "max_top_ranked_per_query": 2,
                    },
                },
            },
        }
    }

    metadata = TaskMetadata(
        type="Any2AnyMultiChoice",
        name="MockMultilingualMultiChoice",
        main_score="accuracy",
        **general_args,  # type: ignore
    )
    metadata.eval_langs = multilingual_eval_langs
    metadata.modalities = ["image", "text"]
    metadata.category = "it2i"

    def load_data(self) -> None:
        images = [self.np_rng.integers(0, 255, (100, 100, 3)) for _ in range(2)]
        images = [
            Image.fromarray(image.astype("uint8")).convert("RGBA") for image in images
        ]

        split_data = RetrievalSplitData(
            queries=Dataset.from_dict(
                {
                    "id": [f"q{i}" for i in range(2)],
                    "image": [images[i] for i in range(2)],
                    "text": [
                        "This is a positive sentence",
                        "This is another positive sentence",
                    ],
                    "modality": ["image,text" for _ in range(2)],
                }
            ),
            corpus=Dataset.from_dict(
                {
                    "id": ["d1", "d2"],
                    "image": [images[i] for i in range(2)],
                    "modality": ["image" for _ in range(2)],
                }
            ),
            relevant_docs={
                "q0": {"d1": 1, "d2": 0},
                "q1": {"d1": 0, "d2": 1},
            },
            top_ranked={
                "q0": ["d1", "d2"],
                "q1": ["d2", "d1"],
            },
        )
        self.dataset = {
            "eng": {
                "test": split_data,
            },
            "fra": {
                "test": split_data,
            },
        }

        self.data_loaded = True


class MockAny2AnyRetrievalI2TTask(AbsTaskRetrieval):
    expected_stats = {
        "test": {
            "num_samples": 4,
            "number_of_characters": 60,
            "documents_text_statistics": {
                "total_text_length": 60,
                "min_text_length": 27,
                "average_text_length": 30.0,
                "max_text_length": 33,
                "unique_texts": 2,
            },
            "documents_image_statistics": None,
            "queries_text_statistics": None,
            "queries_image_statistics": {
                "min_image_width": 100,
                "average_image_width": 100.0,
                "max_image_width": 100,
                "min_image_height": 100,
                "average_image_height": 100.0,
                "max_image_height": 100,
                "unique_images": 2,
            },
            "relevant_docs_statistics": {
                "num_relevant_docs": 2,
                "min_relevant_docs_per_query": 2,
                "average_relevant_docs_per_query": 1.0,
                "max_relevant_docs_per_query": 2,
                "unique_relevant_docs": 2,
            },
            "top_ranked_statistics": None,
        }
    }

    metadata = TaskMetadata(
        type="Any2AnyRetrieval",
        name="MockAny2AnyRetrievalI2T",
        main_score="ndcg_at_10",
        **general_args,  # type: ignore
    )
    metadata.modalities = ["image", "text"]
    metadata.category = "i2t"

    def load_data(self) -> None:
        images = [self.np_rng.integers(0, 255, (100, 100, 3)) for _ in range(2)]
        images = [
            Image.fromarray(image.astype("uint8")).convert("RGBA") for image in images
        ]

        retrieval_split_data = RetrievalSplitData(
            queries=Dataset.from_dict(
                {
                    "id": [f"q{i}" for i in range(2)],
                    "image": [images[i] for i in range(2)],
                    "modality": ["image" for _ in range(2)],
                }
            ),
            corpus=Dataset.from_dict(
                {
                    "id": ["d1", "d2"],
                    "text": [
                        "This is a positive sentence",
                        "This is another positive sentence",
                    ],
                    "modality": ["text" for _ in range(2)],
                }
            ),
            relevant_docs={
                "q0": {"d1": 1, "d2": 0},
                "q1": {"d1": 0, "d2": 1},
            },
            top_ranked=None,
        )
        self.dataset = {"default": {"test": retrieval_split_data}}
        self.data_loaded = True


class MockAny2AnyRetrievalT2ITask(AbsTaskRetrieval):
    expected_stats = {
        "test": {
            "num_samples": 4,
            "number_of_characters": 60,
            "documents_text_statistics": None,
            "documents_image_statistics": {
                "min_image_width": 100,
                "average_image_width": 100.0,
                "max_image_width": 100,
                "min_image_height": 100,
                "average_image_height": 100.0,
                "max_image_height": 100,
                "unique_images": 2,
            },
            "queries_text_statistics": {
                "total_text_length": 60,
                "min_text_length": 27,
                "average_text_length": 30.0,
                "max_text_length": 33,
                "unique_texts": 2,
            },
            "queries_image_statistics": None,
            "relevant_docs_statistics": {
                "num_relevant_docs": 2,
                "min_relevant_docs_per_query": 2,
                "average_relevant_docs_per_query": 1.0,
                "max_relevant_docs_per_query": 2,
                "unique_relevant_docs": 2,
            },
            "top_ranked_statistics": None,
        }
    }

    metadata = TaskMetadata(
        type="Any2AnyRetrieval",
        name="MockAny2AnyRetrievalT2I",
        main_score="ndcg_at_10",
        **general_args,  # type: ignore
    )
    metadata.modalities = ["image", "text"]
    metadata.category = "t2i"

    def load_data(self) -> None:
        images = [self.np_rng.integers(0, 255, (100, 100, 3)) for _ in range(2)]
        images = [
            Image.fromarray(image.astype("uint8")).convert("RGBA") for image in images
        ]

        retrieval_split_data = RetrievalSplitData(
            queries=Dataset.from_dict(
                {
                    "id": [f"q{i}" for i in range(2)],
                    "text": [
                        "This is a positive sentence",
                        "This is another positive sentence",
                    ],
                    "modality": ["text" for _ in range(2)],
                }
            ),
            corpus=Dataset.from_dict(
                {
                    "id": ["d1", "d2"],
                    "image": [images[i] for i in range(2)],
                    "modality": ["image" for _ in range(2)],
                }
            ),
            relevant_docs={
                "q0": {"d1": 1, "d2": 0},
                "q1": {"d1": 0, "d2": 1},
            },
            top_ranked=None,
        )
        self.dataset = {"default": {"test": retrieval_split_data}}
        self.data_loaded = True


class MockImageClassificationTask(AbsTaskClassification):
    expected_stats = {
        "test": {
            "num_samples": 2,
            "number_texts_intersect_with_train": None,
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
            "number_texts_intersect_with_train": None,
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
        **general_args,  # type: ignore
    )
    metadata.modalities = ["image"]
    metadata.category = "i2c"
    n_experiments = 1
    samples_per_label = 5
    input_column_name = "image"

    def load_data(self) -> None:
        images = [self.np_rng.integers(0, 255, (100, 100, 3)) for _ in range(2)]
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


class MockMultilingualImageClassificationTask(AbsTaskClassification):
    n_experiments = 1
    samples_per_label = 5
    expected_stats = {
        "test": {
            "num_samples": 4,
            "number_texts_intersect_with_train": None,
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
                    "number_texts_intersect_with_train": None,
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
                    "number_texts_intersect_with_train": None,
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
            "number_texts_intersect_with_train": None,
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
                    "number_texts_intersect_with_train": None,
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
                    "number_texts_intersect_with_train": None,
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
        **general_args,  # type: ignore
    )
    metadata.modalities = ["image"]
    metadata.category = "i2c"
    metadata.eval_langs = multilingual_eval_langs
    input_column_name = "image"

    def load_data(self) -> None:
        images = [self.np_rng.integers(0, 255, (100, 100, 3)) for _ in range(2)]
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
        **general_args,  # type: ignore
    )
    metadata.modalities = ["image"]
    input_column_name = "image"
    label_column_name = "label"

    def load_data(self) -> None:
        images = [self.np_rng.integers(0, 255, (100, 100, 3)) for _ in range(2)]
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
        **general_args,  # type: ignore
    )
    metadata.modalities = ["image"]
    input_column_name = "image"
    label_column_name = "label"
    max_fraction_of_documents_to_embed = None
    max_document_to_embed = 2

    def load_data(self) -> None:
        images = [self.np_rng.integers(0, 255, (100, 100, 3)) for _ in range(2)]
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


class MockImageMultilabelClassificationTask(AbsTaskMultilabelClassification):
    expected_stats = {
        "test": {
            "num_samples": 4,
            "number_texts_intersect_with_train": None,
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
            "number_texts_intersect_with_train": None,
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
        type="ImageMultilabelClassification",
        name="MockImageMultilabelClassification",
        main_score="accuracy",
        **general_args,  # type: ignore
    )
    metadata.modalities = ["image"]
    metadata.category = "i2c"
    n_experiments = 1
    samples_per_label = 3
    input_column_name = "image"

    def load_data(self) -> None:
        images = [self.np_rng.integers(0, 255, (100, 100, 3)) for _ in range(2)]
        images = [
            Image.fromarray(image.astype("uint8")).convert("RGBA") for image in images
        ]
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


class MockMultilingualImageMultilabelClassificationTask(
    AbsTaskMultilabelClassification
):
    expected_stats = {
        "test": {
            "num_samples": 8,
            "number_texts_intersect_with_train": None,
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
            "label_statistics": {
                "min_labels_per_text": 2,
                "average_label_per_text": 2.0,
                "max_labels_per_text": 2,
                "unique_labels": 4,
                "labels": {
                    "0": {"count": 4},
                    "3": {"count": 4},
                    "1": {"count": 4},
                    "2": {"count": 4},
                },
            },
            "hf_subset_descriptive_stats": {
                "eng": {
                    "num_samples": 4,
                    "number_texts_intersect_with_train": None,
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
                "fra": {
                    "num_samples": 4,
                    "number_texts_intersect_with_train": None,
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
            },
        },
        "train": {
            "num_samples": 20,
            "number_texts_intersect_with_train": None,
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
            "label_statistics": {
                "min_labels_per_text": 2,
                "average_label_per_text": 2.0,
                "max_labels_per_text": 2,
                "unique_labels": 4,
                "labels": {
                    "0": {"count": 10},
                    "3": {"count": 10},
                    "1": {"count": 10},
                    "2": {"count": 10},
                },
            },
            "hf_subset_descriptive_stats": {
                "eng": {
                    "num_samples": 10,
                    "number_texts_intersect_with_train": None,
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
                "fra": {
                    "num_samples": 10,
                    "number_texts_intersect_with_train": None,
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
            },
        },
    }

    metadata = TaskMetadata(
        type="ImageMultilabelClassification",
        name="MockMultilingualImageMultilabelClassification",
        main_score="accuracy",
        **general_args,  # type: ignore
    )
    metadata.modalities = ["image"]
    metadata.eval_langs = multilingual_eval_langs

    input_column_name = "image"

    def load_data(self) -> None:
        images = [self.np_rng.integers(0, 255, (100, 100, 3)) for _ in range(2)]
        images = [
            Image.fromarray(image.astype("uint8")).convert("RGBA") for image in images
        ]
        labels = [["0", "3"], ["1", "2"]]

        data = {
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

        self.dataset = DatasetDict(
            {
                "eng": data,
                "fra": data,
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
        **general_args,  # type: ignore
    )
    metadata.modalities = ["image", "text"]
    metadata.category = "i2t"

    def load_data(self) -> None:
        images = [self.np_rng.integers(0, 255, (100, 100, 3)) for _ in range(2)]
        images = [
            Image.fromarray(image.astype("uint8")).convert("RGBA") for image in images
        ]
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
        **general_args,  # type: ignore
    )
    metadata.modalities = ["image", "text"]
    metadata.category = "i2t"

    metadata.eval_langs = multilingual_eval_langs

    def load_data(self) -> None:
        images = [self.np_rng.integers(0, 255, (100, 100, 3)) for _ in range(2)]
        images = [
            Image.fromarray(image.astype("uint8")).convert("RGBA") for image in images
        ]
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


class MockVisualSTSTask(AbsTaskSTS):
    expected_stats = {
        "test": {
            "num_samples": 2,
            "number_of_characters": None,
            "unique_pairs": None,
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
            "label_statistics": {"min_score": 0.5, "avg_score": 0.5, "max_score": 0.5},
        }
    }

    metadata = TaskMetadata(
        type="VisualSTS(eng)",
        name="MockVisualSTS",
        main_score="cosine_spearman",
        **general_args,  # type: ignore
    )
    metadata.modalities = ["image"]
    metadata.category = "i2i"

    def load_data(self) -> None:
        images = [self.np_rng.integers(0, 255, (100, 100, 3)) for _ in range(2)]

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
        self.dataset = self.dataset.cast_column("sentence1", datasets.Image())
        self.dataset = self.dataset.cast_column("sentence2", datasets.Image())
        self.data_loaded = True


class MockZeroShotClassificationTask(AbsTaskZeroShotClassification):
    expected_stats = {
        "test": {
            "num_samples": 2,
            "number_of_characters": None,
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
            "label_statistics": {
                "min_labels_per_text": 1,
                "average_label_per_text": 1.0,
                "max_labels_per_text": 1,
                "unique_labels": 2,
                "labels": {"label1": {"count": 1}, "label2": {"count": 1}},
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
        **general_args,  # type: ignore
    )
    metadata.modalities = ["image", "text"]
    metadata.category = "i2t"

    def load_data(self) -> None:
        images = [self.np_rng.integers(0, 255, (100, 100, 3)) for _ in range(2)]

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


class MockTextZeroShotClassificationTask(AbsTaskZeroShotClassification):
    expected_stats = {
        "test": {
            "num_samples": 2,
            "number_of_characters": None,
            "text_statistics": {
                "total_text_length": 52,
                "min_text_length": 23,
                "average_text_length": 26.0,
                "max_text_length": 29,
                "unique_texts": 2,
            },
            "image_statistics": None,
            "label_statistics": {
                "min_labels_per_text": 1,
                "average_label_per_text": 1.0,
                "max_labels_per_text": 1,
                "unique_labels": 2,
                "labels": {"label1": {"count": 1}, "label2": {"count": 1}},
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
        **general_args,  # type: ignore
    )
    metadata.modalities = ["text"]
    metadata.category = "t2t"
    input_column_name = "text"

    def load_data(self) -> None:
        texts = ["This is a test sentence", "This is another test sentence"]
        labels = ["label1", "label2"]

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

    def get_candidate_labels(self) -> list[str]:
        return ["This is a test sentence", "This is another test sentence"]


class MockRegressionTask(AbsTaskRegression):
    expected_stats = {
        "test": {
            "num_samples": 2,
            "num_texts_in_train": 1,
            "text_statistics": {
                "total_text_length": 52,
                "min_text_length": 23,
                "average_text_length": 26.0,
                "max_text_length": 29,
                "unique_texts": 2,
            },
            "image_statistics": None,
            "values_statistics": {"min_score": 0.0, "avg_score": 0.5, "max_score": 1.0},
        },
        "train": {
            "num_samples": 2,
            "num_texts_in_train": None,
            "text_statistics": {
                "total_text_length": 53,
                "min_text_length": 23,
                "average_text_length": 26.5,
                "max_text_length": 30,
                "unique_texts": 2,
            },
            "image_statistics": None,
            "values_statistics": {"min_score": 0.0, "avg_score": 0.5, "max_score": 1.0},
        },
    }

    metadata = TaskMetadata(
        type="Regression",
        name="MockRegressionTask",
        main_score="kendalltau",
        **general_args,  # type: ignore
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
            "num_texts_in_train": None,
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
            "values_statistics": {"min_score": 0.0, "avg_score": 0.5, "max_score": 1.0},
        },
        "train": {
            "num_samples": 2,
            "num_texts_in_train": None,
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
            "values_statistics": {"min_score": 0.0, "avg_score": 0.5, "max_score": 1.0},
        },
    }

    metadata = TaskMetadata(
        type="Regression",
        name="MockRegressionTask",
        main_score="kendalltau",
        **general_args,  # type: ignore
    )
    metadata.modalities = ["image"]
    metadata.category = "i2c"
    input_column_name = "image"

    def load_data(self, **kwargs):
        train_images = [self.np_rng.integers(0, 255, (100, 100, 3)) for _ in range(2)]
        train_images = [
            Image.fromarray(image.astype("uint8")).convert("RGBA")
            for image in train_images
        ]

        test_images = [self.np_rng.integers(0, 255, (100, 100, 3)) for _ in range(2)]
        test_images = [
            Image.fromarray(image.astype("uint8")).convert("RGBA")
            for image in test_images
        ]

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
