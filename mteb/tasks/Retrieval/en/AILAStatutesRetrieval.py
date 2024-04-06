from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class AILAStatutes(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="AILAStatutes",
        description="This dataset is structured for the task of identifying the most relevant statutes for a given situation.",
        reference="https://zenodo.org/records/4063986",
        dataset={
            "path": "mteb/AILA_statutes",
            "revision": "ebfcd844eadd3d667efa3c57fc5c8c87f5c2867e",
        },
        type="Retrieval",
        category="p2p",
        eval_splits=["test"],
        eval_langs=["en"],
        main_score="ndcg_at_10",
        date=None,
        form=["written"],
        domains=["Legal"],
        task_subtypes=["Article retrieval"],
        license="CC BY 4.0",
        socioeconomic_status="high",
        annotations_creators="derived",
        dialect=None,
        text_creation="found",
        bibtex_citation= None,
        n_samples=None,
        avg_character_length=None,
    )
