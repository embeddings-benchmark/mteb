from __future__ import annotations

from mteb.abstasks.AbsTaskReranking import AbsTaskReranking
from mteb.abstasks.TaskMetadata import TaskMetadata


class MIRACLFrReranking(AbsTaskReranking):
    metadata = TaskMetadata(
        name="MIRACLFrReranking",
        description="This Reranking Dataset of French queries and sentences",
        reference="https://huggingface.co/datasets/OrdalieTech/MIRACL-FR-Reranking-benchmark",
        dataset={
            "path": "OrdalieTech/MIRACL-FR-Reranking-benchmark",
            "revision": "50bd0db5fa3d5845142fd58d55f870b5bbb27960",
        },
        type="Reranking",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["fra-Latn"],
        main_score="map",
        date=("2023-11-25", "2023-11-27"),  # using dates on commits for date
        form=["written"],
        domains=None,
        task_subtypes=None,
        license="Not specified",
        socioeconomic_status="high",
        annotations_creators="derived",
        dialect=[],
        text_creation="found",
        bibtex_citation=None,
        n_samples={"test": 343},
        avg_character_length=None,
    )
