from __future__ import annotations

from mteb.abstasks.AbsTaskAnyClustering import AbsTaskAnyClustering
from mteb.abstasks.AbsTaskClusteringFast import (
    AbsTaskClusteringFast,
    check_label_distribution,
)
from mteb.abstasks.task_metadata import TaskMetadata


class BiorxivClusteringS2SFast(AbsTaskClusteringFast):
    metadata = TaskMetadata.model_construct(
        name="BiorxivClusteringS2S.v2",
        description="Clustering of titles from biorxiv across 26 categories.",
        reference="https://api.biorxiv.org/",
        dataset={
            "path": "mteb/biorxiv-clustering-s2s",
            "revision": "eb4edb10386758d274cd161093eb351381a16dbf",
        },
        type="Clustering",
        category="t2c",
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
        prompt="Identify the main category of Biorxiv papers based on the titles",
        adapted_from=["BiorxivClusteringS2S"],
    )

    def dataset_transform(self):
        for split in self.metadata.eval_splits:
            check_label_distribution(self.dataset[split])


class BiorxivClusteringS2S(AbsTaskAnyClustering):
    superseded_by = "BiorxivClusteringS2S.v2"
    metadata = TaskMetadata.model_construct(
        name="BiorxivClusteringS2S",
        description="Clustering of titles from biorxiv. Clustering of 10 sets, based on the main category.",
        reference="https://api.biorxiv.org/",
        dataset={
            "path": "mteb/biorxiv-clustering-s2s",
            "revision": "258694dd0231531bc1fd9de6ceb52a0853c6d908",
        },
        type="Clustering",
        category="t2c",
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
        prompt="Identify the main category of Biorxiv papers based on the titles",
    )
