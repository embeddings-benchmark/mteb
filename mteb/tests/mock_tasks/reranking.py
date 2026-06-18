from __future__ import annotations

from typing import TYPE_CHECKING

from datasets import Audio, Dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData
from mteb.abstasks.task_metadata import TaskMetadata

if TYPE_CHECKING:
    pass

from .utils import (
    base_retrieval_datasplit,
    create_mock_audio,
    general_args,
    instruction_retrieval_datasplit,
    multilingual_eval_langs,
)


class MockRerankingTask(AbsTaskRetrieval):
    expected_stats = {
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
        **general_args,
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        base_datasplit = base_retrieval_datasplit()

        self.dataset = {"default": {"test": base_datasplit}}
        self.data_loaded = True


class MockMultilingualRerankingTask(AbsTaskRetrieval):
    expected_stats = {
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
        **general_args,
    )
    metadata.eval_langs = multilingual_eval_langs

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        base_datasplit = base_retrieval_datasplit()
        self.dataset = {
            "eng": {"test": base_datasplit},
            "fra": {"test": base_datasplit},
        }
        self.data_loaded = True


class MockInstructionReranking(AbsTaskRetrieval):
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
        **general_args,
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        base_datasplit = instruction_retrieval_datasplit()
        self.dataset = {"default": {"test": base_datasplit}}
        self.data_loaded = True


class MockMultilingualInstructionReranking(AbsTaskRetrieval):
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
        **general_args,
    )
    metadata.eval_langs = multilingual_eval_langs

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        base_datasplit = instruction_retrieval_datasplit()
        self.dataset = {
            "eng": {"test": base_datasplit, "val": base_datasplit},
            "fra": {"test": base_datasplit, "val": base_datasplit},
        }
        self.data_loaded = True


class MockAudioReranking(AbsTaskRetrieval):
    metadata = TaskMetadata(
        type="AudioReranking",
        name="MockAudioReranking",
        main_score="map_at_1",
        **general_args,  # type: ignore[arg-type]
    )
    metadata.category = "a2a"
    metadata.modalities = ["audio"]

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
            "top_ranked_statistics": {
                "num_top_ranked": 4,
                "min_top_ranked_per_query": 2,
                "average_top_ranked_per_query": 2.0,
                "max_top_ranked_per_query": 2,
            },
        }
    }

    def load_data(self, **kwargs):
        mock_audio = create_mock_audio(self.np_rng)

        queries = Dataset.from_dict(
            {
                "id": ["q1", "q2"],
                "audio": mock_audio,
            }
        )
        corpus = Dataset.from_dict(
            {
                "id": ["d1", "d2"],
                "audio": mock_audio,
            }
        )

        queries = queries.cast_column("audio", Audio())
        corpus = corpus.cast_column("audio", Audio())

        self.dataset = {
            "default": {
                "test": RetrievalSplitData(
                    queries=queries,
                    corpus=corpus,
                    relevant_docs={
                        "q1": {"d1": 1, "d2": 0},
                        "q2": {"d1": 0, "d2": 1},
                    },
                    top_ranked={
                        "q1": ["d1", "d2"],
                        "q2": ["d2", "d1"],
                    },
                )
            }
        }

        self.data_loaded = True
