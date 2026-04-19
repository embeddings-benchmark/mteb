from __future__ import annotations

import random

from mteb.abstasks import AbsTaskPairClassification
from mteb.abstasks.task_metadata import TaskMetadata

from ._video_pair_helpers import build_pair_dataset, generate_pairs


class HumanAnimalCartoonPairClassification(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="HumanAnimalCartoonPairClassification",
        description=(
            "Pair classification on the Human-Animal-Cartoon dataset: "
            "determining whether two videos depict the same action "
            "(e.g. drinking, eating, running) across different actor "
            "domains (human, animal, cartoon)."
        ),
        reference="https://huggingface.co/datasets/mteb/Human-Animal-Cartoon",
        dataset={
            "path": "mteb/Human-Animal-Cartoon",
            "revision": "d38566c4bb055c7325314d3e46610792c2799c4b",
        },
        type="VideoPairClassification",
        category="v2v",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="max_ap",
        date=("2024-01-01", "2024-12-31"),
        domains=["Scene"],
        task_subtypes=["Activity recognition"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video"],
        sample_creation="found",
        bibtex_citation="",
        contributed_by="stef41",
        is_beta=True,
    )

    input1_column_name: str = "video1"
    input2_column_name: str = "video2"
    label_column_name: str = "label"

    def dataset_transform(self, num_proc: int | None = None, **kwargs) -> None:
        rng = random.Random(42)
        for split in self.metadata.eval_splits:
            ds = self.dataset[split]
            pairs = generate_pairs(ds["action"], rng)
            self.dataset[split] = build_pair_dataset(ds, pairs)
