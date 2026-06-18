from __future__ import annotations

from datasets import Audio, Dataset, DatasetDict

from mteb.abstasks.aggregate_task_metadata import AggregateTaskMetadata
from mteb.abstasks.aggregated_task import AbsTaskAggregate
from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData
from mteb.abstasks.task_metadata import TaskMetadata
from mteb.tests.mock_tasks import MockRerankingTask

from .utils import (
    _VIDEO_TEXTS,
    base_retrieval_datasplit,
    create_mock_audio,
    create_mock_images,
    create_mock_video_bytes,
    general_args,
    instruction_retrieval_datasplit,
    multilingual_eval_langs,
)


class MockRetrievalTask(AbsTaskRetrieval):
    _top_k = 2
    expected_stats = {
        "val": {
            "num_samples": 4,
            "num_queries": 2,
            "num_documents": 2,
            "number_of_characters": 136,
            "documents_text_statistics": {
                "total_text_length": 84,
                "min_text_length": 39,
                "average_text_length": 42.0,
                "max_text_length": 45,
                "unique_texts": 2,
            },
            "documents_image_statistics": None,
            "documents_audio_statistics": None,
            "documents_video_statistics": None,
            "queries_text_statistics": {
                "total_text_length": 52,
                "min_text_length": 23,
                "average_text_length": 26.0,
                "max_text_length": 29,
                "unique_texts": 2,
            },
            "queries_image_statistics": None,
            "queries_audio_statistics": None,
            "queries_video_statistics": None,
            "relevant_docs_statistics": {
                "num_relevant_docs": 2,
                "min_relevant_docs_per_query": 1,
                "average_relevant_docs_per_query": 1.0,
                "max_relevant_docs_per_query": 1,
                "unique_relevant_docs": 2,
            },
            "top_ranked_statistics": None,
        },
        "test": {
            "num_samples": 4,
            "num_queries": 2,
            "num_documents": 2,
            "number_of_characters": 136,
            "documents_text_statistics": {
                "total_text_length": 84,
                "min_text_length": 39,
                "average_text_length": 42.0,
                "max_text_length": 45,
                "unique_texts": 2,
            },
            "documents_image_statistics": None,
            "documents_audio_statistics": None,
            "documents_video_statistics": None,
            "queries_text_statistics": {
                "total_text_length": 52,
                "min_text_length": 23,
                "average_text_length": 26.0,
                "max_text_length": 29,
                "unique_texts": 2,
            },
            "queries_image_statistics": None,
            "queries_audio_statistics": None,
            "queries_video_statistics": None,
            "relevant_docs_statistics": {
                "num_relevant_docs": 2,
                "min_relevant_docs_per_query": 1,
                "average_relevant_docs_per_query": 1.0,
                "max_relevant_docs_per_query": 1,
                "unique_relevant_docs": 2,
            },
            "top_ranked_statistics": None,
        },
    }

    metadata = TaskMetadata(
        type="Retrieval",
        name="MockRetrievalTask",
        main_score="ndcg_at_10",
        **dict(general_args | {"eval_splits": ["val", "test"]}),
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        base_datasplit = base_retrieval_datasplit()

        base_datasplit["top_ranked"] = None
        self.dataset = {"default": {"test": base_datasplit, "val": base_datasplit}}
        self.data_loaded = True


class MockRetrievalDialogTask(AbsTaskRetrieval):
    _top_k = 1
    expected_stats = {
        "val": {
            "num_samples": 4,
            "num_queries": 2,
            "num_documents": 2,
            "number_of_characters": 201,
            "documents_text_statistics": {
                "total_text_length": 84,
                "min_text_length": 39,
                "average_text_length": 42.0,
                "max_text_length": 45,
                "unique_texts": 2,
            },
            "documents_image_statistics": None,
            "documents_audio_statistics": None,
            "documents_video_statistics": None,
            "queries_text_statistics": {
                "total_text_length": 117,
                "min_text_length": 37,
                "average_text_length": 58.5,
                "max_text_length": 80,
                "unique_texts": 2,
            },
            "queries_image_statistics": None,
            "queries_audio_statistics": None,
            "queries_video_statistics": None,
            "relevant_docs_statistics": {
                "num_relevant_docs": 2,
                "min_relevant_docs_per_query": 1,
                "average_relevant_docs_per_query": 1.0,
                "max_relevant_docs_per_query": 1,
                "unique_relevant_docs": 2,
            },
            "top_ranked_statistics": None,
        },
        "test": {
            "num_samples": 4,
            "num_queries": 2,
            "num_documents": 2,
            "number_of_characters": 201,
            "documents_text_statistics": {
                "total_text_length": 84,
                "min_text_length": 39,
                "average_text_length": 42.0,
                "max_text_length": 45,
                "unique_texts": 2,
            },
            "documents_image_statistics": None,
            "documents_audio_statistics": None,
            "documents_video_statistics": None,
            "queries_text_statistics": {
                "total_text_length": 117,
                "min_text_length": 37,
                "average_text_length": 58.5,
                "max_text_length": 80,
                "unique_texts": 2,
            },
            "queries_image_statistics": None,
            "queries_audio_statistics": None,
            "queries_video_statistics": None,
            "relevant_docs_statistics": {
                "num_relevant_docs": 2,
                "min_relevant_docs_per_query": 1,
                "average_relevant_docs_per_query": 1.0,
                "max_relevant_docs_per_query": 1,
                "unique_relevant_docs": 2,
            },
            "top_ranked_statistics": None,
        },
    }

    metadata = TaskMetadata(
        type="Retrieval",
        name="MockRetrievalDialogTask",
        main_score="ndcg_at_10",
        **dict(general_args | {"eval_splits": ["val", "test"]}),
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
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
            "num_queries": 4,
            "num_documents": 4,
            "number_of_characters": 272,
            "documents_text_statistics": {
                "total_text_length": 168,
                "min_text_length": 39,
                "average_text_length": 42.0,
                "max_text_length": 45,
                "unique_texts": 2,
            },
            "documents_image_statistics": None,
            "documents_audio_statistics": None,
            "documents_video_statistics": None,
            "queries_text_statistics": {
                "total_text_length": 104,
                "min_text_length": 23,
                "average_text_length": 26.0,
                "max_text_length": 29,
                "unique_texts": 2,
            },
            "queries_image_statistics": None,
            "queries_audio_statistics": None,
            "queries_video_statistics": None,
            "relevant_docs_statistics": {
                "num_relevant_docs": 4,
                "min_relevant_docs_per_query": 1,
                "average_relevant_docs_per_query": 1.0,
                "max_relevant_docs_per_query": 1,
                "unique_relevant_docs": 4,
            },
            "top_ranked_statistics": None,
            "hf_subset_descriptive_stats": {
                "eng": {
                    "num_samples": 4,
                    "num_queries": 2,
                    "num_documents": 2,
                    "number_of_characters": 136,
                    "documents_text_statistics": {
                        "total_text_length": 84,
                        "min_text_length": 39,
                        "average_text_length": 42.0,
                        "max_text_length": 45,
                        "unique_texts": 2,
                    },
                    "documents_image_statistics": None,
                    "documents_audio_statistics": None,
                    "documents_video_statistics": None,
                    "queries_text_statistics": {
                        "total_text_length": 52,
                        "min_text_length": 23,
                        "average_text_length": 26.0,
                        "max_text_length": 29,
                        "unique_texts": 2,
                    },
                    "queries_image_statistics": None,
                    "queries_audio_statistics": None,
                    "queries_video_statistics": None,
                    "relevant_docs_statistics": {
                        "num_relevant_docs": 2,
                        "min_relevant_docs_per_query": 1,
                        "average_relevant_docs_per_query": 1.0,
                        "max_relevant_docs_per_query": 1,
                        "unique_relevant_docs": 2,
                    },
                    "top_ranked_statistics": None,
                },
                "fra": {
                    "num_samples": 4,
                    "num_queries": 2,
                    "num_documents": 2,
                    "number_of_characters": 136,
                    "documents_text_statistics": {
                        "total_text_length": 84,
                        "min_text_length": 39,
                        "average_text_length": 42.0,
                        "max_text_length": 45,
                        "unique_texts": 2,
                    },
                    "documents_image_statistics": None,
                    "documents_audio_statistics": None,
                    "documents_video_statistics": None,
                    "queries_text_statistics": {
                        "total_text_length": 52,
                        "min_text_length": 23,
                        "average_text_length": 26.0,
                        "max_text_length": 29,
                        "unique_texts": 2,
                    },
                    "queries_image_statistics": None,
                    "queries_audio_statistics": None,
                    "queries_video_statistics": None,
                    "relevant_docs_statistics": {
                        "num_relevant_docs": 2,
                        "min_relevant_docs_per_query": 1,
                        "average_relevant_docs_per_query": 1.0,
                        "max_relevant_docs_per_query": 1,
                        "unique_relevant_docs": 2,
                    },
                    "top_ranked_statistics": None,
                },
            },
        },
        "test": {
            "num_samples": 8,
            "num_queries": 4,
            "num_documents": 4,
            "number_of_characters": 272,
            "documents_text_statistics": {
                "total_text_length": 168,
                "min_text_length": 39,
                "average_text_length": 42.0,
                "max_text_length": 45,
                "unique_texts": 2,
            },
            "documents_image_statistics": None,
            "documents_audio_statistics": None,
            "documents_video_statistics": None,
            "queries_text_statistics": {
                "total_text_length": 104,
                "min_text_length": 23,
                "average_text_length": 26.0,
                "max_text_length": 29,
                "unique_texts": 2,
            },
            "queries_image_statistics": None,
            "queries_audio_statistics": None,
            "queries_video_statistics": None,
            "relevant_docs_statistics": {
                "num_relevant_docs": 4,
                "min_relevant_docs_per_query": 1,
                "average_relevant_docs_per_query": 1.0,
                "max_relevant_docs_per_query": 1,
                "unique_relevant_docs": 4,
            },
            "top_ranked_statistics": None,
            "hf_subset_descriptive_stats": {
                "eng": {
                    "num_samples": 4,
                    "num_queries": 2,
                    "num_documents": 2,
                    "number_of_characters": 136,
                    "documents_text_statistics": {
                        "total_text_length": 84,
                        "min_text_length": 39,
                        "average_text_length": 42.0,
                        "max_text_length": 45,
                        "unique_texts": 2,
                    },
                    "documents_image_statistics": None,
                    "documents_audio_statistics": None,
                    "documents_video_statistics": None,
                    "queries_text_statistics": {
                        "total_text_length": 52,
                        "min_text_length": 23,
                        "average_text_length": 26.0,
                        "max_text_length": 29,
                        "unique_texts": 2,
                    },
                    "queries_image_statistics": None,
                    "queries_audio_statistics": None,
                    "queries_video_statistics": None,
                    "relevant_docs_statistics": {
                        "num_relevant_docs": 2,
                        "min_relevant_docs_per_query": 1,
                        "average_relevant_docs_per_query": 1.0,
                        "max_relevant_docs_per_query": 1,
                        "unique_relevant_docs": 2,
                    },
                    "top_ranked_statistics": None,
                },
                "fra": {
                    "num_samples": 4,
                    "num_queries": 2,
                    "num_documents": 2,
                    "number_of_characters": 136,
                    "documents_text_statistics": {
                        "total_text_length": 84,
                        "min_text_length": 39,
                        "average_text_length": 42.0,
                        "max_text_length": 45,
                        "unique_texts": 2,
                    },
                    "documents_image_statistics": None,
                    "documents_audio_statistics": None,
                    "documents_video_statistics": None,
                    "queries_text_statistics": {
                        "total_text_length": 52,
                        "min_text_length": 23,
                        "average_text_length": 26.0,
                        "max_text_length": 29,
                        "unique_texts": 2,
                    },
                    "queries_image_statistics": None,
                    "queries_audio_statistics": None,
                    "queries_video_statistics": None,
                    "relevant_docs_statistics": {
                        "num_relevant_docs": 2,
                        "min_relevant_docs_per_query": 1,
                        "average_relevant_docs_per_query": 1.0,
                        "max_relevant_docs_per_query": 1,
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
        **dict(general_args | {"eval_splits": ["val", "test"]}),
    )
    metadata.eval_langs = multilingual_eval_langs

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        base_datasplit = base_retrieval_datasplit()

        base_datasplit["top_ranked"] = None
        self.dataset = {
            "eng": {"test": base_datasplit, "val": base_datasplit},
            "fra": {"test": base_datasplit, "val": base_datasplit},
        }
        self.data_loaded = True


class MockInstructionRetrieval(AbsTaskRetrieval):
    expected_stats = {
        "test": {
            "num_samples": 4,
            "num_queries": 2,
            "num_documents": 2,
            "number_of_characters": 196,
            "documents_text_statistics": {
                "total_text_length": 84,
                "min_text_length": 39,
                "average_text_length": 42.0,
                "max_text_length": 45,
                "unique_texts": 2,
            },
            "documents_image_statistics": None,
            "documents_audio_statistics": None,
            "documents_video_statistics": None,
            "queries_text_statistics": {
                "total_text_length": 112,
                "min_text_length": 50,
                "average_text_length": 56.0,
                "max_text_length": 62,
                "unique_texts": 2,
            },
            "queries_image_statistics": None,
            "queries_audio_statistics": None,
            "queries_video_statistics": None,
            "relevant_docs_statistics": {
                "num_relevant_docs": 2,
                "min_relevant_docs_per_query": 1,
                "average_relevant_docs_per_query": 1.0,
                "max_relevant_docs_per_query": 1,
                "unique_relevant_docs": 2,
            },
            "top_ranked_statistics": None,
        }
    }

    metadata = TaskMetadata(
        type="InstructionRetrieval",
        name="MockInstructionRetrieval",
        main_score="ndcg_at_10",
        **general_args,
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        base_datasplit = instruction_retrieval_datasplit()
        base_datasplit["top_ranked"] = None

        self.dataset = {"default": {"test": base_datasplit}}
        self.data_loaded = True


class MockMultilingualInstructionRetrieval(AbsTaskRetrieval):
    expected_stats = {
        "test": {
            "num_samples": 8,
            "num_queries": 4,
            "num_documents": 4,
            "number_of_characters": 392,
            "documents_text_statistics": {
                "total_text_length": 168,
                "min_text_length": 39,
                "average_text_length": 42.0,
                "max_text_length": 45,
                "unique_texts": 2,
            },
            "documents_image_statistics": None,
            "documents_audio_statistics": None,
            "documents_video_statistics": None,
            "queries_text_statistics": {
                "total_text_length": 224,
                "min_text_length": 50,
                "average_text_length": 56.0,
                "max_text_length": 62,
                "unique_texts": 2,
            },
            "queries_image_statistics": None,
            "queries_audio_statistics": None,
            "queries_video_statistics": None,
            "relevant_docs_statistics": {
                "num_relevant_docs": 4,
                "min_relevant_docs_per_query": 1,
                "average_relevant_docs_per_query": 1.0,
                "max_relevant_docs_per_query": 1,
                "unique_relevant_docs": 4,
            },
            "top_ranked_statistics": None,
            "hf_subset_descriptive_stats": {
                "eng": {
                    "num_samples": 4,
                    "num_queries": 2,
                    "num_documents": 2,
                    "number_of_characters": 196,
                    "documents_text_statistics": {
                        "total_text_length": 84,
                        "min_text_length": 39,
                        "average_text_length": 42.0,
                        "max_text_length": 45,
                        "unique_texts": 2,
                    },
                    "documents_image_statistics": None,
                    "documents_audio_statistics": None,
                    "documents_video_statistics": None,
                    "queries_text_statistics": {
                        "total_text_length": 112,
                        "min_text_length": 50,
                        "average_text_length": 56.0,
                        "max_text_length": 62,
                        "unique_texts": 2,
                    },
                    "queries_image_statistics": None,
                    "queries_audio_statistics": None,
                    "queries_video_statistics": None,
                    "relevant_docs_statistics": {
                        "num_relevant_docs": 2,
                        "min_relevant_docs_per_query": 1,
                        "average_relevant_docs_per_query": 1.0,
                        "max_relevant_docs_per_query": 1,
                        "unique_relevant_docs": 2,
                    },
                    "top_ranked_statistics": None,
                },
                "fra": {
                    "num_samples": 4,
                    "num_queries": 2,
                    "num_documents": 2,
                    "number_of_characters": 196,
                    "documents_text_statistics": {
                        "total_text_length": 84,
                        "min_text_length": 39,
                        "average_text_length": 42.0,
                        "max_text_length": 45,
                        "unique_texts": 2,
                    },
                    "documents_image_statistics": None,
                    "documents_audio_statistics": None,
                    "documents_video_statistics": None,
                    "queries_text_statistics": {
                        "total_text_length": 112,
                        "min_text_length": 50,
                        "average_text_length": 56.0,
                        "max_text_length": 62,
                        "unique_texts": 2,
                    },
                    "queries_image_statistics": None,
                    "queries_audio_statistics": None,
                    "queries_video_statistics": None,
                    "relevant_docs_statistics": {
                        "num_relevant_docs": 2,
                        "min_relevant_docs_per_query": 1,
                        "average_relevant_docs_per_query": 1.0,
                        "max_relevant_docs_per_query": 1,
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
        **general_args,
    )
    metadata.eval_langs = multilingual_eval_langs

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        base_datasplit = instruction_retrieval_datasplit()
        base_datasplit["top_ranked"] = None
        self.dataset = {
            "eng": {"test": base_datasplit},
            "fra": {"test": base_datasplit},
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
        **general_args,
    )


class MockMultiChoiceTask(AbsTaskRetrieval):
    expected_stats = {
        "test": {
            "num_samples": 4,
            "num_queries": 2,
            "num_documents": 2,
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
            "documents_audio_statistics": None,
            "documents_video_statistics": None,
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
            "queries_audio_statistics": None,
            "queries_video_statistics": None,
            "relevant_docs_statistics": {
                "num_relevant_docs": 2,
                "min_relevant_docs_per_query": 1,
                "average_relevant_docs_per_query": 1.0,
                "max_relevant_docs_per_query": 1,
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
        type="VisionCentricQA",  # TODO: Is this correct?
        name="MockVisionCentricQA",
        main_score="accuracy",
        **general_args,
    )
    metadata.modalities = ["image", "text"]
    metadata.category = "it2i"

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        images = create_mock_images(self.np_rng)
        retrieval_split_data = RetrievalSplitData(
            queries=Dataset.from_dict(
                {
                    "id": [f"q{i}" for i in range(2)],
                    "image": [images[i] for i in range(2)],
                    "text": [
                        "This is a positive sentence",
                        "This is another positive sentence",
                    ],
                }
            ),
            corpus=Dataset.from_dict(
                {
                    "id": ["d1", "d2"],
                    "image": [images[i] for i in range(2)],
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
            "num_queries": 4,
            "num_documents": 4,
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
            "documents_audio_statistics": None,
            "documents_video_statistics": None,
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
            "queries_audio_statistics": None,
            "queries_video_statistics": None,
            "relevant_docs_statistics": {
                "num_relevant_docs": 4,
                "min_relevant_docs_per_query": 1,
                "average_relevant_docs_per_query": 1.0,
                "max_relevant_docs_per_query": 1,
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
                    "num_queries": 2,
                    "num_documents": 2,
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
                    "documents_audio_statistics": None,
                    "documents_video_statistics": None,
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
                    "queries_audio_statistics": None,
                    "queries_video_statistics": None,
                    "relevant_docs_statistics": {
                        "num_relevant_docs": 2,
                        "min_relevant_docs_per_query": 1,
                        "average_relevant_docs_per_query": 1.0,
                        "max_relevant_docs_per_query": 1,
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
                    "num_queries": 2,
                    "num_documents": 2,
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
                    "documents_audio_statistics": None,
                    "documents_video_statistics": None,
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
                    "queries_audio_statistics": None,
                    "queries_video_statistics": None,
                    "relevant_docs_statistics": {
                        "num_relevant_docs": 2,
                        "min_relevant_docs_per_query": 1,
                        "average_relevant_docs_per_query": 1.0,
                        "max_relevant_docs_per_query": 1,
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
        type="VisionCentricQA",
        name="MockMultilingualVisionCentricQA",
        main_score="accuracy",
        **general_args,
    )
    metadata.eval_langs = multilingual_eval_langs
    metadata.modalities = ["image", "text"]
    metadata.category = "it2i"

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        images = create_mock_images(self.np_rng)

        split_data = RetrievalSplitData(
            queries=Dataset.from_dict(
                {
                    "id": [f"q{i}" for i in range(2)],
                    "image": [images[i] for i in range(2)],
                    "text": [
                        "This is a positive sentence",
                        "This is another positive sentence",
                    ],
                }
            ),
            corpus=Dataset.from_dict(
                {
                    "id": ["d1", "d2"],
                    "image": [images[i] for i in range(2)],
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
            "num_queries": 2,
            "num_documents": 2,
            "number_of_characters": 60,
            "documents_text_statistics": {
                "total_text_length": 60,
                "min_text_length": 27,
                "average_text_length": 30.0,
                "max_text_length": 33,
                "unique_texts": 2,
            },
            "documents_image_statistics": None,
            "documents_audio_statistics": None,
            "documents_video_statistics": None,
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
            "queries_audio_statistics": None,
            "queries_video_statistics": None,
            "relevant_docs_statistics": {
                "num_relevant_docs": 2,
                "min_relevant_docs_per_query": 1,
                "average_relevant_docs_per_query": 1.0,
                "max_relevant_docs_per_query": 1,
                "unique_relevant_docs": 2,
            },
            "top_ranked_statistics": None,
        }
    }

    metadata = TaskMetadata(
        type="Any2AnyRetrieval",
        name="MockAny2AnyRetrievalI2T",
        main_score="ndcg_at_10",
        **general_args,
    )
    metadata.modalities = ["image", "text"]
    metadata.category = "i2t"

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        images = create_mock_images(self.np_rng)

        retrieval_split_data = RetrievalSplitData(
            queries=Dataset.from_dict(
                {
                    "id": [f"q{i}" for i in range(2)],
                    "image": [images[i] for i in range(2)],
                }
            ),
            corpus=Dataset.from_dict(
                {
                    "id": ["d1", "d2"],
                    "text": [
                        "This is a positive sentence",
                        "This is another positive sentence",
                    ],
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
            "num_queries": 2,
            "num_documents": 2,
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
            "documents_audio_statistics": None,
            "documents_video_statistics": None,
            "queries_text_statistics": {
                "total_text_length": 60,
                "min_text_length": 27,
                "average_text_length": 30.0,
                "max_text_length": 33,
                "unique_texts": 2,
            },
            "queries_image_statistics": None,
            "queries_audio_statistics": None,
            "queries_video_statistics": None,
            "relevant_docs_statistics": {
                "num_relevant_docs": 2,
                "min_relevant_docs_per_query": 1,
                "average_relevant_docs_per_query": 1.0,
                "max_relevant_docs_per_query": 1,
                "unique_relevant_docs": 2,
            },
            "top_ranked_statistics": None,
        }
    }
    metadata = TaskMetadata(
        type="Any2AnyRetrieval",
        name="MockAny2AnyRetrievalT2I",
        main_score="ndcg_at_10",
        **general_args,
    )
    metadata.modalities = ["image", "text"]
    metadata.category = "t2i"

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        images = create_mock_images(self.np_rng)

        retrieval_split_data = RetrievalSplitData(
            queries=Dataset.from_dict(
                {
                    "id": [f"q{i}" for i in range(2)],
                    "text": [
                        "This is a positive sentence",
                        "This is another positive sentence",
                    ],
                }
            ),
            corpus=Dataset.from_dict(
                {
                    "id": ["d1", "d2"],
                    "image": [images[i] for i in range(2)],
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


class MockAny2AnyRetrievalT2ATask(AbsTaskRetrieval):
    metadata = TaskMetadata(
        type="Any2AnyRetrieval",
        name="MockAny2AnyRetrievalT2A",
        main_score="ndcg_at_10",
        **general_args,  # type: ignore[arg-type]
    )
    metadata.modalities = ["audio", "text"]
    metadata.category = "t2a"
    expected_stats = {
        "test": {
            "num_samples": 4,
            "num_queries": 2,
            "num_documents": 2,
            "number_of_characters": 60,
            "documents_text_statistics": None,
            "documents_image_statistics": None,
            "documents_audio_statistics": {
                "total_duration_seconds": 2.0,
                "min_duration_seconds": 1.0,
                "average_duration_seconds": 1.0,
                "max_duration_seconds": 1.0,
                "unique_audios": 2,
                "average_sampling_rate": 16000.0,
                "sampling_rates": {16000: 2},
            },
            "documents_video_statistics": None,
            "queries_text_statistics": {
                "total_text_length": 60,
                "min_text_length": 27,
                "average_text_length": 30.0,
                "max_text_length": 33,
                "unique_texts": 2,
            },
            "queries_image_statistics": None,
            "queries_audio_statistics": None,
            "queries_video_statistics": None,
            "relevant_docs_statistics": {
                "num_relevant_docs": 2,
                "min_relevant_docs_per_query": 1,
                "average_relevant_docs_per_query": 1.0,
                "max_relevant_docs_per_query": 1,
                "unique_relevant_docs": 2,
            },
            "top_ranked_statistics": None,
        }
    }

    def load_data(self, **kwargs):
        mock_audio = create_mock_audio(self.np_rng)

        self.queries = DatasetDict(
            {
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
        )
        self.corpus = DatasetDict(
            {
                "test": Dataset.from_dict(
                    {
                        "id": ["d1", "d2"],
                        "audio": mock_audio,
                        "modality": ["audio" for _ in range(2)],
                    }
                )
            }
        )
        self.corpus = self.corpus.cast_column("audio", Audio())

        self.relevant_docs = {
            "test": {
                "q0": {"d1": 1, "d2": 0},
                "q1": {"d1": 0, "d2": 1},
            },
        }
        self.data_loaded = True


class MockAny2AnyRetrievalA2TTask(AbsTaskRetrieval):
    metadata = TaskMetadata(
        type="Any2AnyRetrieval",
        name="MockAny2AnyRetrievalA2T",
        main_score="ndcg_at_10",
        **general_args,  # type: ignore[arg-type]
    )
    metadata.modalities = ["audio", "text"]
    metadata.category = "a2t"

    expected_stats = {
        "test": {
            "num_samples": 4,
            "num_queries": 2,
            "num_documents": 2,
            "number_of_characters": 60,
            "documents_text_statistics": {
                "total_text_length": 60,
                "min_text_length": 27,
                "average_text_length": 30.0,
                "max_text_length": 33,
                "unique_texts": 2,
            },
            "documents_image_statistics": None,
            "documents_audio_statistics": None,
            "documents_video_statistics": None,
            "queries_text_statistics": None,
            "queries_image_statistics": None,
            "queries_audio_statistics": {
                "total_duration_seconds": 2.0,
                "min_duration_seconds": 1.0,
                "average_duration_seconds": 1.0,
                "max_duration_seconds": 1.0,
                "unique_audios": 2,
                "average_sampling_rate": 16000.0,
                "sampling_rates": {16000: 2},
            },
            "queries_video_statistics": None,
            "relevant_docs_statistics": {
                "num_relevant_docs": 2,
                "min_relevant_docs_per_query": 1,
                "average_relevant_docs_per_query": 1.0,
                "max_relevant_docs_per_query": 1,
                "unique_relevant_docs": 2,
            },
            "top_ranked_statistics": None,
        }
    }

    def load_data(self, **kwargs):
        mock_audio = create_mock_audio(self.np_rng)

        self.queries = DatasetDict(
            {
                "test": Dataset.from_dict(
                    {
                        "id": [f"q{i}" for i in range(2)],
                        "audio": mock_audio,
                        "modality": ["audio" for _ in range(2)],
                    }
                )
            }
        )
        self.queries = self.queries.cast_column("audio", Audio())

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


class MockAny2AnyRetrievalA2ATask(AbsTaskRetrieval):
    metadata = TaskMetadata(
        type="Any2AnyRetrieval",
        name="MockAny2AnyRetrievalA2A",
        main_score="ndcg_at_10",
        **general_args,  # type: ignore[arg-type]
    )
    metadata.modalities = ["audio"]
    metadata.category = "a2a"

    expected_stats = {
        "test": {
            "num_samples": 4,
            "num_queries": 2,
            "num_documents": 2,
            "number_of_characters": 0,
            "documents_text_statistics": None,
            "documents_image_statistics": None,
            "documents_audio_statistics": {
                "total_duration_seconds": 2.0,
                "min_duration_seconds": 1.0,
                "average_duration_seconds": 1.0,
                "max_duration_seconds": 1.0,
                "unique_audios": 2,
                "average_sampling_rate": 16000.0,
                "sampling_rates": {16000: 2},
            },
            "documents_video_statistics": None,
            "queries_text_statistics": None,
            "queries_image_statistics": None,
            "queries_audio_statistics": {
                "total_duration_seconds": 2.0,
                "min_duration_seconds": 1.0,
                "average_duration_seconds": 1.0,
                "max_duration_seconds": 1.0,
                "unique_audios": 2,
                "average_sampling_rate": 16000.0,
                "sampling_rates": {16000: 2},
            },
            "queries_video_statistics": None,
            "relevant_docs_statistics": {
                "num_relevant_docs": 2,
                "min_relevant_docs_per_query": 1,
                "average_relevant_docs_per_query": 1.0,
                "max_relevant_docs_per_query": 1,
                "unique_relevant_docs": 2,
            },
            "top_ranked_statistics": None,
        }
    }

    def load_data(self, **kwargs):
        mock_audio = create_mock_audio(self.np_rng)

        self.queries = DatasetDict(
            {
                "test": Dataset.from_dict(
                    {
                        "id": [f"q{i}" for i in range(2)],
                        "audio": mock_audio,
                        "modality": ["audio" for _ in range(2)],
                    }
                )
            }
        )
        self.corpus = DatasetDict(
            {
                "test": Dataset.from_dict(
                    {
                        "id": ["d1", "d2"],
                        "audio": mock_audio,
                        "modality": ["audio" for _ in range(2)],
                    }
                )
            }
        )

        self.queries = self.queries.cast_column("audio", Audio())
        self.corpus = self.corpus.cast_column("audio", Audio())

        self.relevant_docs = {
            "test": {
                "q0": {"d1": 1, "d2": 0},
                "q1": {"d1": 0, "d2": 1},
            },
        }
        self.data_loaded = True


class MockVideoRetrievalV2T(AbsTaskRetrieval):
    """Video queries → text corpus (v2t)"""

    metadata = TaskMetadata(
        type="Any2AnyRetrieval",
        name="MockVideoRetrievalV2T",
        main_score="ndcg_at_10",
        **general_args,  # type: ignore[arg-type]
    )
    metadata.modalities = ["video", "text"]
    metadata.category = "v2t"

    expected_stats = {
        "test": {
            "num_samples": 4,
            "num_queries": 2,
            "num_documents": 2,
            "number_of_characters": 60,
            "documents_text_statistics": {
                "total_text_length": 60,
                "min_text_length": 28,
                "average_text_length": 30.0,
                "max_text_length": 32,
                "unique_texts": 2,
            },
            "documents_image_statistics": None,
            "documents_audio_statistics": None,
            "documents_video_statistics": None,
            "queries_text_statistics": None,
            "queries_image_statistics": None,
            "queries_audio_statistics": None,
            "queries_video_statistics": {
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
            "relevant_docs_statistics": {
                "num_relevant_docs": 2,
                "min_relevant_docs_per_query": 1,
                "average_relevant_docs_per_query": 1.0,
                "max_relevant_docs_per_query": 1,
                "unique_relevant_docs": 2,
            },
            "top_ranked_statistics": None,
        }
    }

    def load_data(self, **kwargs):
        from datasets import Video

        mock_videos = create_mock_video_bytes(self.np_rng)

        queries = Dataset.from_dict(
            {
                "id": [f"q{i}" for i in range(2)],
                "video": mock_videos,
                "modality": ["video", "video"],
            }
        )
        queries = queries.cast_column("video", Video())

        corpus = Dataset.from_dict(
            {
                "id": ["d1", "d2"],
                "text": _VIDEO_TEXTS,
                "modality": ["text", "text"],
            }
        )

        relevant_docs = {
            "q0": {"d1": 1, "d2": 0},
            "q1": {"d1": 0, "d2": 1},
        }
        self.dataset = {
            "default": {
                "test": RetrievalSplitData(
                    queries=queries,
                    corpus=corpus,
                    relevant_docs=relevant_docs,
                    top_ranked=None,
                )
            }
        }
        self.data_loaded = True


class MockVideoRetrievalT2V(AbsTaskRetrieval):
    """Text queries → video corpus (t2v)"""

    metadata = TaskMetadata(
        type="Any2AnyRetrieval",
        name="MockVideoRetrievalT2V",
        main_score="ndcg_at_10",
        **general_args,  # type: ignore[arg-type]
    )
    metadata.modalities = ["text", "video"]
    metadata.category = "t2v"

    expected_stats = {
        "test": {
            "num_samples": 4,
            "num_queries": 2,
            "num_documents": 2,
            "number_of_characters": 60,
            "documents_text_statistics": None,
            "documents_image_statistics": None,
            "documents_audio_statistics": None,
            "documents_video_statistics": {
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
            "queries_text_statistics": {
                "total_text_length": 60,
                "min_text_length": 28,
                "average_text_length": 30.0,
                "max_text_length": 32,
                "unique_texts": 2,
            },
            "queries_image_statistics": None,
            "queries_audio_statistics": None,
            "queries_video_statistics": None,
            "relevant_docs_statistics": {
                "num_relevant_docs": 2,
                "min_relevant_docs_per_query": 1,
                "average_relevant_docs_per_query": 1.0,
                "max_relevant_docs_per_query": 1,
                "unique_relevant_docs": 2,
            },
            "top_ranked_statistics": None,
        }
    }

    def load_data(self, **kwargs):
        from datasets import Video

        mock_videos = create_mock_video_bytes(self.np_rng)

        queries = Dataset.from_dict(
            {
                "id": [f"q{i}" for i in range(2)],
                "text": _VIDEO_TEXTS,
                "modality": ["text", "text"],
            }
        )

        corpus = Dataset.from_dict(
            {
                "id": ["d1", "d2"],
                "video": mock_videos,
                "modality": ["video", "video"],
            }
        )
        corpus = corpus.cast_column("video", Video())

        relevant_docs = {
            "q0": {"d1": 1, "d2": 0},
            "q1": {"d1": 0, "d2": 1},
        }
        self.dataset = {
            "default": {
                "test": RetrievalSplitData(
                    queries=queries,
                    corpus=corpus,
                    relevant_docs=relevant_docs,
                    top_ranked=None,
                )
            }
        }
        self.data_loaded = True


class MockVideoAudioRetrievalVA2T(AbsTaskRetrieval):
    """Video+audio queries → text corpus (va2t)"""

    metadata = TaskMetadata(
        type="Any2AnyRetrieval",
        name="MockVideoAudioRetrievalVA2T",
        main_score="ndcg_at_10",
        **general_args,  # type: ignore[arg-type]
    )
    metadata.modalities = ["video", "audio", "text"]
    metadata.category = "va2t"

    expected_stats = {
        "test": {
            "num_samples": 4,
            "num_queries": 2,
            "num_documents": 2,
            "number_of_characters": 60,
            "documents_text_statistics": {
                "total_text_length": 60,
                "min_text_length": 28,
                "average_text_length": 30.0,
                "max_text_length": 32,
                "unique_texts": 2,
            },
            "documents_image_statistics": None,
            "documents_audio_statistics": None,
            "documents_video_statistics": None,
            "queries_text_statistics": None,
            "queries_image_statistics": None,
            "queries_audio_statistics": {
                "total_duration_seconds": 2.0,
                "min_duration_seconds": 1.0,
                "average_duration_seconds": 1.0,
                "max_duration_seconds": 1.0,
                "unique_audios": 2,
                "average_sampling_rate": 16000.0,
                "sampling_rates": {16000: 2},
            },
            "queries_video_statistics": {
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
            "relevant_docs_statistics": {
                "num_relevant_docs": 2,
                "min_relevant_docs_per_query": 1,
                "average_relevant_docs_per_query": 1.0,
                "max_relevant_docs_per_query": 1,
                "unique_relevant_docs": 2,
            },
            "top_ranked_statistics": None,
        }
    }

    def load_data(self, **kwargs):
        from datasets import Video

        mock_videos = create_mock_video_bytes(self.np_rng)
        mock_audio = create_mock_audio(self.np_rng)

        queries = Dataset.from_dict(
            {
                "id": [f"q{i}" for i in range(2)],
                "video": mock_videos,
                "audio": mock_audio,
                "modality": ["video_audio", "video_audio"],
            }
        )
        queries = queries.cast_column("video", Video())
        queries = queries.cast_column("audio", Audio())

        corpus = Dataset.from_dict(
            {
                "id": ["d1", "d2"],
                "text": _VIDEO_TEXTS,
                "modality": ["text", "text"],
            }
        )

        relevant_docs = {
            "q0": {"d1": 1, "d2": 0},
            "q1": {"d1": 0, "d2": 1},
        }
        self.dataset = {
            "default": {
                "test": RetrievalSplitData(
                    queries=queries,
                    corpus=corpus,
                    relevant_docs=relevant_docs,
                    top_ranked=None,
                )
            }
        }
        self.data_loaded = True


class MockVideoAudioRetrievalT2VA(AbsTaskRetrieval):
    """Text queries → video+audio corpus (t2va)"""

    metadata = TaskMetadata(
        type="Any2AnyRetrieval",
        name="MockVideoAudioRetrievalT2VA",
        main_score="ndcg_at_10",
        **general_args,  # type: ignore[arg-type]
    )
    metadata.modalities = ["text", "video", "audio"]
    metadata.category = "t2va"

    expected_stats = {
        "test": {
            "num_samples": 4,
            "num_queries": 2,
            "num_documents": 2,
            "number_of_characters": 60,
            "documents_text_statistics": None,
            "documents_image_statistics": None,
            "documents_audio_statistics": {
                "total_duration_seconds": 2.0,
                "min_duration_seconds": 1.0,
                "average_duration_seconds": 1.0,
                "max_duration_seconds": 1.0,
                "unique_audios": 2,
                "average_sampling_rate": 16000.0,
                "sampling_rates": {16000: 2},
            },
            "documents_video_statistics": {
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
            "queries_text_statistics": {
                "total_text_length": 60,
                "min_text_length": 28,
                "average_text_length": 30.0,
                "max_text_length": 32,
                "unique_texts": 2,
            },
            "queries_image_statistics": None,
            "queries_audio_statistics": None,
            "queries_video_statistics": None,
            "relevant_docs_statistics": {
                "num_relevant_docs": 2,
                "min_relevant_docs_per_query": 1,
                "average_relevant_docs_per_query": 1.0,
                "max_relevant_docs_per_query": 1,
                "unique_relevant_docs": 2,
            },
            "top_ranked_statistics": None,
        }
    }

    def load_data(self, **kwargs):
        from datasets import Video

        mock_videos = create_mock_video_bytes(self.np_rng)
        mock_audio = create_mock_audio(self.np_rng)

        queries = Dataset.from_dict(
            {
                "id": [f"q{i}" for i in range(2)],
                "text": _VIDEO_TEXTS,
                "modality": ["text", "text"],
            }
        )

        corpus = Dataset.from_dict(
            {
                "id": ["d1", "d2"],
                "video": mock_videos,
                "audio": mock_audio,
                "modality": ["video_audio", "video_audio"],
            }
        )
        corpus = corpus.cast_column("video", Video())
        corpus = corpus.cast_column("audio", Audio())

        relevant_docs = {
            "q0": {"d1": 1, "d2": 0},
            "q1": {"d1": 0, "d2": 1},
        }
        self.dataset = {
            "default": {
                "test": RetrievalSplitData(
                    queries=queries,
                    corpus=corpus,
                    relevant_docs=relevant_docs,
                    top_ranked=None,
                )
            }
        }
        self.data_loaded = True


class MockVideoAudioTextRetrievalVAT2T(AbsTaskRetrieval):
    """Video+audio+text queries → text corpus (vat2t)"""

    metadata = TaskMetadata(
        type="Any2AnyRetrieval",
        name="MockVideoAudioTextRetrievalVAT2T",
        main_score="ndcg_at_10",
        **general_args,  # type: ignore[arg-type]
    )
    metadata.modalities = ["video", "audio", "text"]
    metadata.category = "vat2t"

    expected_stats = {
        "test": {
            "num_samples": 4,
            "num_queries": 2,
            "num_documents": 2,
            "number_of_characters": 120,
            "documents_text_statistics": {
                "total_text_length": 60,
                "min_text_length": 28,
                "average_text_length": 30.0,
                "max_text_length": 32,
                "unique_texts": 2,
            },
            "documents_image_statistics": None,
            "documents_audio_statistics": None,
            "documents_video_statistics": None,
            "queries_text_statistics": {
                "total_text_length": 60,
                "min_text_length": 28,
                "average_text_length": 30.0,
                "max_text_length": 32,
                "unique_texts": 2,
            },
            "queries_image_statistics": None,
            "queries_audio_statistics": {
                "total_duration_seconds": 2.0,
                "min_duration_seconds": 1.0,
                "average_duration_seconds": 1.0,
                "max_duration_seconds": 1.0,
                "unique_audios": 2,
                "average_sampling_rate": 16000.0,
                "sampling_rates": {16000: 2},
            },
            "queries_video_statistics": {
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
            "relevant_docs_statistics": {
                "num_relevant_docs": 2,
                "min_relevant_docs_per_query": 1,
                "average_relevant_docs_per_query": 1.0,
                "max_relevant_docs_per_query": 1,
                "unique_relevant_docs": 2,
            },
            "top_ranked_statistics": None,
        }
    }

    def load_data(self, **kwargs):
        from datasets import Video

        mock_videos = create_mock_video_bytes(self.np_rng)
        mock_audio = create_mock_audio(self.np_rng)

        queries = Dataset.from_dict(
            {
                "id": [f"q{i}" for i in range(2)],
                "video": mock_videos,
                "audio": mock_audio,
                "text": _VIDEO_TEXTS,
                "modality": ["video_audio_text", "video_audio_text"],
            }
        )
        queries = queries.cast_column("video", Video())
        queries = queries.cast_column("audio", Audio())

        corpus = Dataset.from_dict(
            {
                "id": ["d1", "d2"],
                "text": _VIDEO_TEXTS,
                "modality": ["text", "text"],
            }
        )
        relevant_docs = {
            "q0": {"d1": 1, "d2": 0},
            "q1": {"d1": 0, "d2": 1},
        }
        self.dataset = {
            "default": {
                "test": RetrievalSplitData(
                    queries=queries,
                    corpus=corpus,
                    relevant_docs=relevant_docs,
                    top_ranked=None,
                )
            }
        }

        self.data_loaded = True
