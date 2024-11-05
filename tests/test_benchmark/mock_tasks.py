"""This implements minimal viable mock tasks for testing the benchmarking framework."""

from __future__ import annotations

from datasets import Dataset, DatasetDict

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
            "average_text_length": 26.0,
            "unique_labels": 2,
            "labels": {"0": {"count": 1}, "1": {"count": 1}},
        }
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
            "average_text_length": 26.0,
            "unique_labels": 2,
            "labels": {"0": {"count": 2}, "1": {"count": 2}},
            "hf_subset_descriptive_stats": {
                "eng": {
                    "num_samples": 2,
                    "number_of_characters": 52,
                    "average_text_length": 26.0,
                    "unique_labels": 2,
                    "labels": {"0": {"count": 1}, "1": {"count": 1}},
                },
                "fra": {
                    "num_samples": 2,
                    "number_of_characters": 52,
                    "average_text_length": 26.0,
                    "unique_labels": 2,
                    "labels": {"0": {"count": 1}, "1": {"count": 1}},
                },
            },
        }
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
            "average_sentence1_length": 26.0,
            "average_sentence2_length": 30.5,
            "num_samples": 2,
            "number_of_characters": 113,
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
            "average_sentence1_length": 26.0,
            "average_sentence2_length": 30.5,
            "num_samples": 4,
            "number_of_characters": 226,
            "hf_subset_descriptive_stats": {
                "eng": {
                    "average_sentence1_length": 26.0,
                    "average_sentence2_length": 30.5,
                    "num_samples": 2,
                    "number_of_characters": 113,
                },
                "fra": {
                    "average_sentence1_length": 26.0,
                    "average_sentence2_length": 30.5,
                    "num_samples": 2,
                    "number_of_characters": 113,
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
            "average_sentence1_length": 28.25,
            "average_sentence2_length": 28.25,
            "num_samples": 4,
            "number_of_characters": 226,
            "hf_subset_descriptive_stats": {
                "eng_Latn-fra_Latn": {
                    "average_sentence1_length": 26.0,
                    "average_sentence2_length": 30.5,
                    "num_samples": 2,
                    "number_of_characters": 113,
                },
                "fra_Latn-eng_Latn": {
                    "average_sentence1_length": 30.5,
                    "average_sentence2_length": 26.0,
                    "num_samples": 2,
                    "number_of_characters": 113,
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
            "average_text_length": 3.0,
            "average_labels_per_text": 3.0,
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
            "average_text_length": 3.0,
            "average_labels_per_text": 3.0,
            "unique_labels": 3,
            "labels": {"0": {"count": 2}, "1": {"count": 2}, "2": {"count": 2}},
            "hf_subset_descriptive_stats": {
                "eng": {
                    "num_samples": 1,
                    "number_of_characters": 3,
                    "average_text_length": 3.0,
                    "average_labels_per_text": 3.0,
                    "unique_labels": 3,
                    "labels": {"0": {"count": 1}, "1": {"count": 1}, "2": {"count": 1}},
                },
                "fra": {
                    "num_samples": 1,
                    "number_of_characters": 3,
                    "average_text_length": 3.0,
                    "average_labels_per_text": 3.0,
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
            "average_text_length": 27.0,
            "average_labels_per_text": 1.0,
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
            "average_text_length": 27.0,
            "average_labels_per_text": 1.0,
            "unique_labels": 3,
            "labels": {"0": {"count": 2}, "1": {"count": 2}, "2": {"count": 2}},
            "hf_subset_descriptive_stats": {
                "eng": {
                    "num_samples": 3,
                    "number_of_characters": 81,
                    "average_text_length": 27.0,
                    "average_labels_per_text": 1.0,
                    "unique_labels": 3,
                    "labels": {"0": {"count": 1}, "1": {"count": 1}, "2": {"count": 1}},
                },
                "fra": {
                    "num_samples": 3,
                    "number_of_characters": 81,
                    "average_text_length": 27.0,
                    "average_labels_per_text": 1.0,
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
            "avg_sentence1_len": 26.0,
            "avg_sentence2_len": 30.5,
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
            "avg_sentence1_len": 26.0,
            "avg_sentence2_len": 30.5,
            "unique_labels": 2,
            "labels": {"1": {"count": 2}, "0": {"count": 2}},
            "hf_subset_descriptive_stats": {
                "eng": {
                    "num_samples": 2,
                    "number_of_characters": 113,
                    "avg_sentence1_len": 26.0,
                    "avg_sentence2_len": 30.5,
                    "unique_labels": 2,
                    "labels": {"1": {"count": 1}, "0": {"count": 1}},
                },
                "fra": {
                    "num_samples": 2,
                    "number_of_characters": 113,
                    "avg_sentence1_len": 26.0,
                    "avg_sentence2_len": 30.5,
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
            "average_sentence1_len": 26.0,
            "average_sentence2_len": 30.5,
            "avg_score": 0.5,
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
            "average_sentence1_len": 26.0,
            "average_sentence2_len": 30.5,
            "avg_score": 0.5,
            "hf_subset_descriptive_stats": {
                "eng": {
                    "num_samples": 2,
                    "number_of_characters": 113,
                    "average_sentence1_len": 26.0,
                    "average_sentence2_len": 30.5,
                    "avg_score": 0.5,
                },
                "fra": {
                    "num_samples": 2,
                    "number_of_characters": 113,
                    "average_sentence1_len": 26.0,
                    "average_sentence2_len": 30.5,
                    "avg_score": 0.5,
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
            "avg_text_len": 26.0,
            "avg_human_summaries_len": 2.0,
            "avg_machine_summaries_len": 2.0,
            "avg_relevance": 0.5,
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
            "avg_text_len": 26.0,
            "avg_human_summaries_len": 2.0,
            "avg_machine_summaries_len": 2.0,
            "avg_relevance": 0.5,
            "hf_subset_descriptive_stats": {
                "eng": {
                    "num_samples": 2,
                    "number_of_characters": 60,
                    "avg_text_len": 26.0,
                    "avg_human_summaries_len": 2.0,
                    "avg_machine_summaries_len": 2.0,
                    "avg_relevance": 0.5,
                },
                "fra": {
                    "num_samples": 2,
                    "number_of_characters": 60,
                    "avg_text_len": 26.0,
                    "avg_human_summaries_len": 2.0,
                    "avg_machine_summaries_len": 2.0,
                    "avg_relevance": 0.5,
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
            "avg_query_len": 26.0,
            "avg_positive_len": 30.0,
            "avg_negative_len": 30.0,
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
            "avg_query_len": 26.0,
            "avg_positive_len": 30.0,
            "avg_negative_len": 30.0,
            "hf_subset_descriptive_stats": {
                "eng": {
                    "num_samples": 2,
                    "number_of_characters": 172,
                    "num_positive": 2,
                    "num_negative": 2,
                    "avg_query_len": 26.0,
                    "avg_positive_len": 30.0,
                    "avg_negative_len": 30.0,
                },
                "fra": {
                    "num_samples": 2,
                    "number_of_characters": 172,
                    "num_positive": 2,
                    "num_negative": 2,
                    "avg_query_len": 26.0,
                    "avg_positive_len": 30.0,
                    "avg_negative_len": 30.0,
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
            "num_samples": 4,
            "number_of_characters": 56.0,
            "average_document_length": 15.0,
            "average_query_length": 13.0,
            "num_documents": 2,
            "num_queries": 2,
            "average_relevant_docs_per_query": 1.0,
        }
    }

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
        self.data_loaded = True


class MockMultilingualRetrievalTask(AbsTaskRetrieval, MultilingualTask):
    expected_stats = {
        "test": {
            "number_of_characters": 56.0,
            "num_samples": 8,
            "num_queries": 4,
            "num_documents": 4,
            "average_document_length": 7.5,
            "average_query_length": 6.5,
            "average_relevant_docs_per_query": 1.0,
            "hf_subset_descriptive_stats": {
                "eng": {
                    "number_of_characters": 56.0,
                    "num_samples": 4,
                    "num_queries": 2,
                    "num_documents": 2,
                    "average_document_length": 15.0,
                    "average_query_length": 13.0,
                    "average_relevant_docs_per_query": 1.0,
                },
                "fra": {
                    "number_of_characters": 56.0,
                    "num_samples": 4,
                    "num_queries": 2,
                    "num_documents": 2,
                    "average_document_length": 15.0,
                    "average_query_length": 13.0,
                    "average_relevant_docs_per_query": 1.0,
                },
            },
        }
    }

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
        self.data_loaded = True


class MockMultilabelClassification(AbsTaskMultilabelClassification):
    expected_stats = {
        "test": {
            "average_text_length": 26.0,
            "number_of_characters": 156,
            "average_label_per_text": 2.0,
            "num_samples": 6,
            "unique_labels": 2,
            "labels": {"0": {"count": 6}, "1": {"count": 6}},
        }
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
            "average_text_length": 26.0,
            "number_of_characters": 312,
            "average_label_per_text": 2.0,
            "num_samples": 12,
            "unique_labels": 2,
            "labels": {"0": {"count": 12}, "1": {"count": 12}},
            "hf_subset_descriptive_stats": {
                "eng": {
                    "average_text_length": 26.0,
                    "number_of_characters": 156,
                    "average_label_per_text": 2.0,
                    "num_samples": 6,
                    "unique_labels": 2,
                    "labels": {"0": {"count": 6}, "1": {"count": 6}},
                },
                "fra": {
                    "average_text_length": 26.0,
                    "number_of_characters": 156,
                    "average_label_per_text": 2.0,
                    "num_samples": 6,
                    "unique_labels": 2,
                    "labels": {"0": {"count": 6}, "1": {"count": 6}},
                },
            },
        }
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


class MockInstructionRetrival(AbsTaskInstructionRetrieval):
    do_length_ablation = True
    expected_stats = {
        "test": {
            "average_changed_instruction_length": 37.0,
            "average_document_length": 30.0,
            "average_instruction_length": 29.0,
            "average_query_length": 26.0,
            "average_relevant_docs_per_query": 1.0,
            "average_top_ranked_per_query": 2.0,
            "num_docs": 2,
            "num_queries": 2,
            "num_samples": 4,
            "number_of_characters": 244,
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
            "average_document_length": 30.0,
            "average_query_length": 26.0,
            "average_instruction_length": 29.0,
            "average_changed_instruction_length": 37.0,
            "average_relevant_docs_per_query": 1.0,
            "average_top_ranked_per_query": 2.0,
            "hf_subset_descriptive_stats": {
                "eng": {
                    "num_samples": 4,
                    "num_docs": 2,
                    "num_queries": 2,
                    "number_of_characters": 244,
                    "average_document_length": 30.0,
                    "average_query_length": 26.0,
                    "average_instruction_length": 29.0,
                    "average_changed_instruction_length": 37.0,
                    "average_relevant_docs_per_query": 1.0,
                    "average_top_ranked_per_query": 2.0,
                },
                "fra": {
                    "num_samples": 4,
                    "num_docs": 2,
                    "num_queries": 2,
                    "number_of_characters": 244,
                    "average_document_length": 30.0,
                    "average_query_length": 26.0,
                    "average_instruction_length": 29.0,
                    "average_changed_instruction_length": 37.0,
                    "average_relevant_docs_per_query": 1.0,
                    "average_top_ranked_per_query": 2.0,
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
