from __future__ import annotations

import itertools

from datasets import Dataset, DatasetDict

from mteb.abstasks.AbsTaskClustering import AbsTaskClustering
from mteb.abstasks.AbsTaskClusteringFast import (
    AbsTaskClusteringFast,
    check_label_distribution,
)
from mteb.abstasks.TaskMetadata import TaskMetadata


class MedrxivClusteringP2PFast(AbsTaskClusteringFast):
    metadata = TaskMetadata(
        name="MedrxivClusteringP2P.v2",
        description="Clustering of titles+abstract from medrxiv across 51 categories.",
        reference="https://api.medrxiv.org/",
        dataset={
            "path": "mteb/medrxiv-clustering-p2p",
            "revision": "e7a26af6f3ae46b30dde8737f02c07b1505bcc73",
        },
        type="Clustering",
        category="p2p",
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
        prompt="Identify the main category of Medrxiv papers based on the titles and abstracts",
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


class MedrxivClusteringP2P(AbsTaskClustering):
    superseded_by = "MedrxivClusteringP2P.v2"
    metadata = TaskMetadata(
        name="MedrxivClusteringP2P",
        description="Clustering of titles+abstract from medrxiv. Clustering of 10 sets, based on the main category.",
        reference="https://api.medrxiv.org/",
        dataset={
            "path": "mteb/medrxiv-clustering-p2p",
            "revision": "e7a26af6f3ae46b30dde8737f02c07b1505bcc73",
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
        license="https://www.medrxiv.org/content/about-medrxiv",
        annotations_creators="derived",
        dialect=[],
        sample_creation="created",
        bibtex_citation="",
        prompt="Identify the main category of Medrxiv papers based on the titles and abstracts",
    )
