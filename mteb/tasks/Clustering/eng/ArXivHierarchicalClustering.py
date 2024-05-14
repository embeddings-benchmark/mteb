from __future__ import annotations

import itertools

from datasets import Dataset, DatasetDict

from mteb.abstasks.AbsTaskClusteringFast import AbsTaskClusteringFast
from mteb.abstasks.TaskMetadata import TaskMetadata


def split_labels(record: dict) -> dict:
    record["labels"] = record["labels"].split(".")
    return record


class ArXivHierarchicalClusteringP2P(AbsTaskClusteringFast):
    metadata = TaskMetadata(
        name="ArXivHierarchicalClusteringP2P",
        description="Clustering of titles+abstract from arxiv. Clustering of 30 sets, either on the main or secondary category",
        reference="https://www.kaggle.com/Cornell-University/arxiv",
        dataset={
            "path": "mteb/arxiv-clustering-p2p",
            "revision": "0bbdb47bcbe3a90093699aefeed338a0f28a7ee8",
        },
        type="Clustering",
        category="p2p",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="v_measure",
        date=("1991-01-01", "2021-01-01"),  # 1991-01-01 is the first arxiv paper
        form=["written"],
        domains=["Academic"],
        task_subtypes=[],
        license="CC0",
        socioeconomic_status="high",
        annotations_creators="derived",
        dialect=["Thematic clustering"],
        text_creation="found",
        bibtex_citation="@misc{arXiv.org e-Print archive, url={https://arxiv.org/} }",
        n_samples={"test": 250_000},
        avg_character_length={"test": 1009.98},
    )

    def dataset_transform(self):
        ds = dict()
        for split in self.metadata.eval_splits:
            labels = itertools.chain.from_iterable(self.dataset[split]["labels"])
            sentences = itertools.chain.from_iterable(self.dataset[split]["sentences"])
            ds[split] = Dataset.from_dict({"labels": labels, "sentences": sentences})
        self.dataset = DatasetDict(ds)
        self.dataset = self.dataset.map(split_labels)
        self.dataset = self.stratified_subsampling(
            self.dataset,
            self.seed,
            self.metadata.eval_splits,
            label="labels",
            n_samples=2048,
        )


class ArXivHierarchicalClusteringS2S(AbsTaskClusteringFast):
    metadata = TaskMetadata(
        name="ArXivHierarchicalClusteringS2S",
        description="Clustering of titles from arxiv. Clustering of 30 sets, either on the main or secondary category",
        reference="https://www.kaggle.com/Cornell-University/arxiv",
        dataset={
            "path": "mteb/arxiv-clustering-s2s",
            "revision": "b73bd54100e5abfa6e3a23dcafb46fe4d2438dc3",
        },
        type="Clustering",
        category="p2p",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="v_measure",
        date=("1991-01-01", "2021-01-01"),  # 1991-01-01 is the first arxiv paper
        form=["written"],
        domains=["Academic"],
        task_subtypes=["Thematic clustering"],
        license="CC0",
        socioeconomic_status="high",
        annotations_creators="derived",
        dialect=[],
        text_creation="found",
        bibtex_citation="@misc{arXiv.org e-Print archive, url={https://arxiv.org/} }",
        n_samples={"test": 250_000},
        avg_character_length={"test": 1009.98},
    )

    def dataset_transform(self):
        ds = dict()
        for split in self.metadata.eval_splits:
            labels = itertools.chain.from_iterable(self.dataset[split]["labels"])
            sentences = itertools.chain.from_iterable(self.dataset[split]["sentences"])
            ds[split] = Dataset.from_dict({"labels": labels, "sentences": sentences})
        self.dataset = DatasetDict(ds)
        self.dataset = self.dataset.map(split_labels)
        self.dataset = self.stratified_subsampling(
            self.dataset,
            self.seed,
            self.metadata.eval_splits,
            label="labels",
            n_samples=2048,
        )
