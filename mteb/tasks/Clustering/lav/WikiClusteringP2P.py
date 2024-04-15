from __future__ import annotations

from mteb.abstasks.AbsTaskClustering import AbsTaskClustering
from mteb.abstasks.TaskMetadata import TaskMetadata


class BlurbsClusteringP2P(AbsTaskClustering):
    metadata = TaskMetadata(
        name="WikiClusteringP2P",
        description="Clustering of wikipedia articles inspired by BlubrbsClusteringP2P. Labels are taken from top-level categories as defined in https://lv.wikipedia.org/wiki/Kategorija:Pamatkategorijas.",
        reference=None,
        dataset={
            "path": "ryzzlestrizzle/lvwiki-clustering-p2p",
            "revision": "9cc393f46b4e7136870d9438921720c75451d34f",
        },
        type="Clustering",
        category="p2p",
        eval_splits=["test"],
        eval_langs=["lav-Latn"],
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
        avg_character_length={"test": 749.37},
    )
