from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskClustering import AbsTaskClustering


class TenKGnadClusteringP2P(AbsTaskClustering):
    metadata = TaskMetadata(
        name="TenKGnadClusteringP2P",
        description="Clustering of news article titles+subheadings+texts. Clustering of 10 splits on the news article category.",
        reference="https://tblock.github.io/10kGNAD/",
        dataset={
            "path": "slvnwhrl/tenkgnad-clustering-p2p",
            "revision": "5c59e41555244b7e45c9a6be2d720ab4bafae558",
        },
        type="Clustering",
        category="p2p",
        eval_splits=["test"],
        eval_langs=["deu-Latn"],
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
        n_samples={"test": 45914},
        avg_character_length={"test": 2641.03},
    )
