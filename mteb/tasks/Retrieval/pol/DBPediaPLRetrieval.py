from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class DBPediaPL(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="DBPedia-PL",
        description="DBpedia-Entity is a standard test collection for entity search over the DBpedia knowledge base",
        reference="https://github.com/iai-group/DBpedia-Entity/",
        dataset={
            "path": "clarin-knext/dbpedia-pl",
            "revision": "76afe41d9af165cc40999fcaa92312b8b012064a",
        },
        type="Retrieval",
        category="s2p",
        eval_splits=["test"],
        eval_langs=["pol-Latn"],
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
