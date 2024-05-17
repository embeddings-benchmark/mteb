from __future__ import annotations

import itertools

import numpy as np
from datasets import Dataset, DatasetDict

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskClustering import AbsTaskClustering
from ....abstasks.AbsTaskClusteringFast import AbsTaskClusteringFast


class StackExchangeClusteringP2PFast(AbsTaskClusteringFast):
    metadata = TaskMetadata(
        name="StackExchangeClusteringP2PFast",
        description="Clustering of title+body from stackexchange. Clustering of 5 sets of 10k paragraphs and 5 sets of 5k paragraphs.",
        reference="https://arxiv.org/abs/2104.07081",
        dataset={
            "path": "mteb/stackexchange-clustering-p2p",
            "revision": "815ca46b2622cec33ccafc3735d572c266efdb44",
        },
        type="Clustering",
        category="p2p",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
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
        n_samples={"test": 16000},
        avg_character_length={"test": 1090.7},
    )

    def dataset_transform(self):
        ds = dict()
        for split in self.metadata.eval_splits:
            labels = list(itertools.chain.from_iterable(self.dataset[split]["labels"]))
            sentences = list(
                itertools.chain.from_iterable(self.dataset[split]["sentences"])
            )

            # Remove sentences and labels with only 1 label example.
            unique_labels, counts = np.unique(labels, return_counts=True)
            solo_label_idx = np.where(counts == 1)
            solo_labels = unique_labels[solo_label_idx]
            for solo_label in solo_labels:
                loc = labels.index(solo_label)
                labels.pop(loc)
                sentences.pop(loc)
            ds[split] = Dataset.from_dict({"labels": labels, "sentences": sentences})

        self.dataset = DatasetDict(ds)
        self.dataset = self.stratified_subsampling(
            self.dataset,
            self.seed,
            self.metadata.eval_splits,
            label="labels",
            n_samples=16000,
        )


class StackExchangeClusteringP2P(AbsTaskClustering):
    superseeded_by = "StackExchangeClusteringP2PFast"
    metadata = TaskMetadata(
        name="StackExchangeClusteringP2P",
        description="Clustering of title+body from stackexchange. Clustering of 5 sets of 10k paragraphs and 5 sets of 5k paragraphs.",
        reference="https://arxiv.org/abs/2104.07081",
        dataset={
            "path": "mteb/stackexchange-clustering-p2p",
            "revision": "815ca46b2622cec33ccafc3735d572c266efdb44",
        },
        type="Clustering",
        category="p2p",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
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
        n_samples={"test": 75000},
        avg_character_length={"test": 1090.7},
    )
