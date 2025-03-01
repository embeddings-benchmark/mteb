from __future__ import annotations

from mteb.abstasks.AbsTaskClustering import AbsTaskClustering
from mteb.abstasks.AbsTaskClusteringFast import (
    AbsTaskClusteringFast,
    check_label_distribution,
)
from mteb.abstasks.TaskMetadata import TaskMetadata


class BiorxivClusteringP2PFast(AbsTaskClusteringFast):
    metadata = TaskMetadata(
        name="BiorxivClusteringP2P.v2",
        description="Clustering of titles+abstract from biorxiv across 26 categories.",
        reference="https://api.biorxiv.org/",
        dataset={
            "path": "mteb/biorxiv-clustering-p2p",
            "revision": "f5dbc242e11dd8e24def4c4268607a49e02946dc",
        },
        type="Clustering",
        category="p2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="v_measure",
        date=("2021-01-01", "2022-05-10"),
        domains=["Academic", "Written"],
        task_subtypes=["Thematic clustering"],
        license="https://www.biorxiv.org/content/about-biorxiv",
        annotations_creators="derived",
        dialect=[],
        sample_creation="created",
        bibtex_citation="",
        prompt="Identify the main category of Biorxiv papers based on the titles and abstracts",
        adapted_from=["BiorxivClusteringP2P"],
    )

    def dataset_transform(self):
        for split in self.metadata.eval_splits:
            check_label_distribution(self.dataset[split])


class BiorxivClusteringP2P(AbsTaskClustering):
    superseded_by = "BiorxivClusteringP2P.v2"
    metadata = TaskMetadata(
        name="BiorxivClusteringP2P",
        description="Clustering of titles+abstract from biorxiv. Clustering of 10 sets, based on the main category.",
        reference="https://api.biorxiv.org/",
        dataset={
            "path": "mteb/biorxiv-clustering-p2p",
            "revision": "65b79d1d13f80053f67aca9498d9402c2d9f1f40",
        },
        type="Clustering",
        category="p2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="v_measure",
        date=("2021-01-01", "2022-05-10"),
        domains=["Academic", "Written"],
        task_subtypes=["Thematic clustering"],
        license="https://www.biorxiv.org/content/about-biorxiv",
        annotations_creators="derived",
        dialect=[],
        sample_creation="created",
        bibtex_citation="",
        prompt="Identify the main category of Biorxiv papers based on the titles and abstracts",
    )
