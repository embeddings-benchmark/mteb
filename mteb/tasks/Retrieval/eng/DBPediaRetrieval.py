from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class DBPedia(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="DBPedia",
        description="DBpedia-Entity is a standard test collection for entity search over the DBpedia knowledge base",
        reference="https://github.com/iai-group/DBpedia-Entity/",
        dataset={
            "path": "mteb/dbpedia",
            "revision": "c0f706b76e590d620bd6618b3ca8efdd34e2d659",
        },
        type="Retrieval",
        category="s2p",
        eval_splits=["dev", "test"],
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
