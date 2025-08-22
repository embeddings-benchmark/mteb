from __future__ import annotations

from mteb.abstasks.Image.AbsTaskZeroShotClassification import (
    AbsTaskZeroShotClassification,
)
from mteb.abstasks.TaskMetadata import TaskMetadata


class RenderedSST2(AbsTaskZeroShotClassification):
    metadata = TaskMetadata(
        name="RenderedSST2",
        description="RenderedSST2.",
        reference="https://huggingface.co/datasets/clip-benchmark/wds_renderedsst2",
        dataset={
            "path": "clip-benchmark/wds_renderedsst2",
            "revision": "66b9a461eda025201dd147e5f390f5984c33643a",
        },
        type="ZeroShotClassification",
        category="i2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2016-01-01", "2016-12-31"),
        domains=["Reviews"],
        task_subtypes=[],
        license="mit",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="created",
        bibtex_citation="",
        descriptive_stats={
            "n_samples": {"test": 1820},
            "avg_character_length": {"test": 10.0},
        },
    )

    # Override default column names in the subclass
    image_column_name: str = "png"
    label_column_name: str = "cls"

    def get_candidate_labels(self) -> list[str]:
        return ["a negative review of a movie", "a positive review of a movie"]
