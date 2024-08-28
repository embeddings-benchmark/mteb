from __future__ import annotations

import itertools

from datasets import Dataset, DatasetDict

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
            "revision": "35191c8c0dca72d8ff3efcd72aa802307d469663",
        },
        type="Clustering",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="v_measure",
        date=("2021-01-01", "2022-05-10"),
        domains=["Academic", "Medical", "Written"],
        task_subtypes=["Thematic clustering"],
        license="https://www.medrxiv.org/content/about-medrxiv",
        annotations_creators="derived",
        dialect=[],
        sample_creation="created",
        bibtex_citation="",
        descriptive_stats={
            "n_samples": {"test": 1500},
            "avg_character_length": {"test": 114.9},
        },
    )

    def dataset_transform(self):
        ds = {}
        for split in self.metadata.eval_splits:
            labels = list(itertools.chain.from_iterable(self.dataset[split]["labels"]))
            sentences = list(
                itertools.chain.from_iterable(self.dataset[split]["sentences"])
            )
            check_label_distribution(self.dataset[split])
            ds[split] = Dataset.from_dict({"labels": labels, "sentences": sentences})
        self.dataset = DatasetDict(ds)


class MedrxivClusteringS2S(AbsTaskClustering):
    superseded_by = "MedrxivClusteringS2S.v2"
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
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="v_measure",
        date=("2021-01-01", "2022-05-10"),
        domains=["Academic", "Medical", "Written"],
        task_subtypes=["Thematic clustering"],
        license="https://www.medrxiv.org/content/about-medrxiv",
        annotations_creators="derived",
        dialect=[],
        sample_creation="created",
        bibtex_citation="",
        descriptive_stats={
            "n_samples": {"test": 37500},
            "avg_character_length": {"test": 114.7},
        },
    )
