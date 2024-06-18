from __future__ import annotations

from mteb.abstasks.AbsTaskClustering import AbsTaskClustering
from mteb.abstasks.AbsTaskClusteringFast import (
    AbsTaskClusteringFast,
    check_label_distribution,
)
from mteb.abstasks.TaskMetadata import TaskMetadata


class MedrxivClusteringS2SFast(AbsTaskClusteringFast):
    metadata = TaskMetadata(
        name="MedrxivClusteringS2S.v2",
        description="Clustering of titles from medrxiv across 51 categories.",
        reference="https://api.medrxiv.org/",
        dataset={
            "path": "mteb/medrxiv-clustering-s2s",
            "revision": "ec20c81676a749c0f06fb4a9397fc7e168521458",
        },
        type="Clustering",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="v_measure",
        date=("2021-01-01", "2022-05-10"),
        form=["written"],
        domains=["Academic", "Medical"],
        task_subtypes=["Thematic clustering"],
        license="https://www.medrxiv.org/content/about-medrxiv",
        socioeconomic_status="high",
        annotations_creators="derived",
        dialect=[],
        text_creation="created",
        bibtex_citation="",
        n_samples={"test": 2048},
        avg_character_length={"test": 114.9},
    )

    def dataset_transform(self):
        for split in self.metadata.eval_splits:
            check_label_distribution(self.dataset[split])
        self.dataset = self.stratified_subsampling(
            self.dataset,
            self.seed,
            self.metadata.eval_splits,
            label="labels",
            n_samples=2048,
        )


class MedrxivClusteringS2S(AbsTaskClustering):
    superseeded_by = "MedrxivClusteringS2S.v2"
    metadata = TaskMetadata(
        name="MedrxivClusteringS2S",
        description="Clustering of titles from medrxiv. Clustering of 10 sets, based on the main category.",
        reference="https://api.medrxiv.org/",
        dataset={
            "path": "mteb/medrxiv-clustering-s2s",
            "revision": "35191c8c0dca72d8ff3efcd72aa802307d469663",
        },
        type="Clustering",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="v_measure",
        date=("2021-01-01", "2022-05-10"),
        form=["written"],
        domains=["Academic", "Medical"],
        task_subtypes=["Thematic clustering"],
        license="https://www.medrxiv.org/content/about-medrxiv",
        socioeconomic_status="high",
        annotations_creators="derived",
        dialect=[],
        text_creation="created",
        bibtex_citation="",
        n_samples={"test": 375000},
        avg_character_length={"test": 114.7},
    )
