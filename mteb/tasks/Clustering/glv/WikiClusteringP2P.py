from __future__ import annotations

from mteb.abstasks.AbsTaskClustering import AbsTaskClustering
from mteb.abstasks.TaskMetadata import TaskMetadata


class WikiClusteringP2P(AbsTaskClustering):
    metadata = TaskMetadata(
        name="WikiClusteringP2P",
        description="Clustering of wikipedia articles inspired by BlubrbsClusteringP2P. Labels are taken from top-level categories as defined in https://gv.wikipedia.org/wiki/Ronney:Bunneydagh.",
        reference=None,
        dataset={
            "path": "ryzzlestrizzle/gvwiki-clustering-p2p",
            "revision": "2b10ee05f615e32eb20bde8bc4c37c4ad6d7ca2b",
        },
        type="Clustering",
        category="p2p",
        eval_splits=["test"],
        eval_langs=["glv-Latn"],
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
        avg_character_length={"test": 379.92},
    )
