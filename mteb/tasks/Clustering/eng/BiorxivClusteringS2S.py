from __future__ import annotations

from mteb.abstasks.AbsTaskClustering import AbsTaskClustering
from mteb.abstasks.AbsTaskClusteringFast import (
    AbsTaskClusteringFast,
    check_label_distribution,
)
from mteb.abstasks.TaskMetadata import TaskMetadata


class BiorxivClusteringS2SFast(AbsTaskClusteringFast):
    metadata = TaskMetadata(
        name="BiorxivClusteringS2S.v2",
        description="Clustering of titles from biorxiv across 26 categories.",
        reference="https://api.biorxiv.org/",
        dataset={
            "path": "mteb/biorxiv-clustering-s2s",
            "revision": "eb4edb10386758d274cd161093eb351381a16dbf",
        },
        type="Clustering",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="v_measure",
        date=("2021-01-01", "2022-05-10"),
        form=["written"],
        domains=["Academic"],
        task_subtypes=["Thematic clustering"],
        license="https://www.biorxiv.org/content/about-biorxiv",
        socioeconomic_status="high",
        annotations_creators="derived",
        dialect=[],
        text_creation="created",
        bibtex_citation="",
        n_samples={"test": 2151},
        avg_character_length={"test": 101.7},
    )

    def dataset_transform(self):
        for split in self.metadata.eval_splits:
            check_label_distribution(self.dataset[split])


class BiorxivClusteringS2S(AbsTaskClustering):
    superseeded_by = "BiorxivClusteringS2S.v2"
    metadata = TaskMetadata(
        name="BiorxivClusteringS2S",
        description="Clustering of titles from biorxiv. Clustering of 10 sets, based on the main category.",
        reference="https://api.biorxiv.org/",
        dataset={
            "path": "mteb/biorxiv-clustering-s2s",
            "revision": "258694dd0231531bc1fd9de6ceb52a0853c6d908",
        },
        type="Clustering",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="v_measure",
        date=("2021-01-01", "2022-05-10"),
        form=["written"],
        domains=["Academic"],
        task_subtypes=["Thematic clustering"],
        license="https://www.biorxiv.org/content/about-biorxiv",
        socioeconomic_status="high",
        annotations_creators="derived",
        dialect=[],
        text_creation="created",
        bibtex_citation="",
        n_samples={"test": 75000},
        avg_character_length={"test": 101.6},
    )
