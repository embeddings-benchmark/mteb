from __future__ import annotations

from typing import Any

from datasets import Dataset, DatasetDict

from mteb.abstasks.analogy import AbsTaskAnalogy
from mteb.abstasks.task_metadata import TaskMetadata

from .create_mock_samples import create_mock_images

general_args = {
    "description": "A lightweight mock analogy task designed for testing, debugging, and local model verification within the MTEB framework.",
    "reference": "https://github.com/embeddings-benchmark/mteb",
    "dataset": {
        "path": "NA",
        "revision": "NA",
    },
    "type": "Analogy",
    "eval_splits": ["test"],
    "eval_langs": ["eng-Latn"],
    "main_score": "hit_rate_at_10",
    "date": ("2022-12-22", "2022-12-22"),
    "dialect": [],
    "domains": [],
    "task_subtypes": [],
    "license": "cc-by-4.0",
    "annotations_creators": "derived",
    "sample_creation": "found",
    "bibtex_citation": "",
}

_analogies = Dataset.from_dict(
    {
        "word1": ["man", "big", "king"],
        "word2": ["king", "bigger", "queen"],
        "word3": ["woman", "small", "man"],
        "word4": ["queen", "smaller", "woman"],
    }
)


class MockAnalogyTask(AbsTaskAnalogy):
    expected_stats = {
        "test": {
            "num_samples": 3,
            "num_concepts": 8,
            "num_documents": 8,
            "number_of_characters": 38,
            "min_documents_per_concept": 1,
            "average_documents_per_concept": 1.0,
            "max_documents_per_concept": 1,
            "documents_text_statistics": {
                "total_text_length": 38,
                "min_text_length": 3,
                "average_text_length": 4.75,
                "max_text_length": 7,
                "unique_texts": 8,
            },
            "documents_image_statistics": None,
            "documents_audio_statistics": None,
            "documents_video_statistics": None,
        }
    }

    metadata = TaskMetadata(
        name="MockAnalogyTask",
        category="t2t",
        modalities=["text"],
        **general_args,
    )

    def load_data(self, **kwargs: Any) -> None:
        self.dataset = DatasetDict({"test": _analogies})
        self.data_loaded = True


class MockImageAnalogyTask(AbsTaskAnalogy):
    expected_stats = {
        "test": {
            "num_samples": 3,
            "num_concepts": 8,
            "num_documents": 16,
            "number_of_characters": None,
            "min_documents_per_concept": 2,
            "average_documents_per_concept": 2.0,
            "max_documents_per_concept": 2,
            "documents_text_statistics": None,
            "documents_image_statistics": {
                "min_image_width": 100,
                "average_image_width": 100.0,
                "max_image_width": 100,
                "min_image_height": 100,
                "average_image_height": 100.0,
                "max_image_height": 100,
                "unique_images": 16,
                "black_or_white_images": 0,
            },
            "documents_audio_statistics": None,
            "documents_video_statistics": None,
        }
    }

    metadata = TaskMetadata(
        name="MockImageAnalogyTask",
        category="i2i",
        modalities=["image"],
        **general_args,
    )
    input_column_name = "image"

    def load_data(self, **kwargs: Any) -> None:
        self.dataset = DatasetDict({"test": _analogies})
        concepts = sorted(
            {
                word
                for column in self.analogy_column_names
                for word in _analogies[column]
            }
        )
        self.concepts = Dataset.from_dict(
            {
                "concept": [concept for concept in concepts for _ in range(2)],
                "image": create_mock_images(self.np_rng, n=2 * len(concepts)),
            }
        )
        self.data_loaded = True

    def get_concepts(self, analogies: Dataset, split: str) -> Dataset:
        return self.concepts
