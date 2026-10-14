from __future__ import annotations

from typing import Any

from datasets import Dataset, DatasetDict

from mteb.abstasks.spatial_arrangement import AbsTaskSpatialArrangement
from mteb.abstasks.task_metadata import TaskMetadata

# Positions of each text on a shared reference canvas. Each mock arrangement places a subset of the texts at a scaled
# and shifted copy of these positions, so a model that reproduces this layout agrees with every arrangement.
MOCK_CANVAS = {
    "Children need more teachers": (0.0, 0.0),
    "Schools lack funding": (1.0, 0.0),
    "Teachers are overworked": (3.0, 1.0),
    "Hospitals have long waiting lists": (0.0, 4.0),
    "Elderly care is understaffed": (6.0, 6.0),
    "Nurses leave the profession": (2.0, 7.0),
}


def _arrangement(
    texts: list[str], user_id: str, scale: float, shift: tuple[float, float]
) -> dict[str, Any]:
    return {
        "texts": texts,
        "x": [MOCK_CANVAS[text][0] * scale + shift[0] for text in texts],
        "y": [MOCK_CANVAS[text][1] * scale + shift[1] for text in texts],
        "user_id": user_id,
    }


class MockSpatialArrangementTask(AbsTaskSpatialArrangement):
    expected_stats = {
        "test": {
            "num_samples": 2,
            "number_of_characters": 262,
            "num_raters": 2,
            "num_triplets": 60,
            "arranged_texts_statistics": {
                "total_text_length": 262,
                "min_text_length": 20,
                "average_text_length": 26.2,
                "max_text_length": 33,
                "unique_texts": 6,
            },
        }
    }

    metadata = TaskMetadata(
        type="SpatialArrangement",
        name="MockSpatialArrangementTask",
        main_score="alpha_auc",
        description="A lightweight mock spatial arrangement task designed for testing, debugging, and local model verification within the MTEB framework.",
        reference="https://github.com/embeddings-benchmark/mteb",
        dataset={
            "path": "NA",
            "revision": "NA",
        },
        category="t2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        date=("2022-12-22", "2022-12-22"),
        dialect=["Written"],
        domains=[],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        modalities=["text"],
        sample_creation="created",
        bibtex_citation="",
    )

    def load_data(self, num_proc: int | None = None, **kwargs: Any) -> None:
        texts = list(MOCK_CANVAS)
        arrangements = [
            _arrangement(texts[:5], user_id="A", scale=100, shift=(0, 0)),
            _arrangement(texts[1:], user_id="B", scale=50, shift=(300, 200)),
        ]
        self.dataset = DatasetDict({"test": Dataset.from_list(arrangements)})
        self.data_loaded = True
