from __future__ import annotations

from mteb.abstasks.AbsTaskClustering import AbsTaskClustering
from mteb.abstasks.TaskMetadata import TaskMetadata


class WikiClusteringP2P(AbsTaskClustering):
    metadata = TaskMetadata(
        name="WikiClusteringP2P",
        description="Clustering of wikipedia articles inspired by BlubrbsClusteringP2P. Labels are taken from top-level categories as defined in https://sq.wikipedia.org/wiki/Kategoria:Klasifikimi_kryesor_i_temave.",
        reference=None,
        dataset={
            "path": "ryzzlestrizzle/sqwiki-clustering-p2p",
            "revision": "5086259301369cc69d3e6fd4d11e1d31fdbbc7a7",
        },
        type="Clustering",
        category="p2p",
        eval_splits=["test"],
        eval_langs=["sqi-Latn"],
        main_score="v_measure",
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
        n_samples={"test": 150_000},
        avg_character_length={"test": 395.88},
    )
