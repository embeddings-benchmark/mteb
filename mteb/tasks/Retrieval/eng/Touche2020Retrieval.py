from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class Touche2020(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Touche2020",
        description="Touché Task 1: Argument Retrieval for Controversial Questions",
        reference="https://webis.de/events/touche-20/shared-task-1.html",
        dataset={
            "path": "mteb/touche2020",
            "revision": "a34f9a33db75fa0cbb21bb5cfc3dae8dc8bec93f",
        },
        type="Retrieval",
        category="s2p",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=None,
        form=None,
        domains=None,
        task_subtypes=None,
        license=None,
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=None,
        text_creation=None,
        bibtex_citation=None,
        n_samples=None,
        avg_character_length=None,
    )
