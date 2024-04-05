from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class AILACasedocs(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="AILACasedocs",
        description="The task is to retrieve the case document that most closely matches or is most relevant to the scenario described in the provided query.",
        reference="https://zenodo.org/records/4063986",
        dataset={
            "path": "mteb/AILA_casedocs",
            "revision": "4106e6bcc72e0698d714ea8b101355e3e238431a",
        },
        type="Retrieval",
        category="s2p",
        eval_splits=["test"],
        eval_langs=["en"],
        main_score="ndcg_at_10",
        date=None,
        form="written",
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
