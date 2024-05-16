from __future__ import annotations
import itertools

from datasets import Dataset, DatasetDict

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskClusteringFast import AbsTaskClusteringFast


class StackExchangeClustering(AbsTaskClusteringFast):
    metadata = TaskMetadata(
        name="StackExchangeClustering",
        description="Clustering of titles from 121 stackexchanges. Clustering of 25 sets, each with 10-50 classes, and each class with 100 - 1000 sentences.",
        reference="https://arxiv.org/abs/2104.07081",
        dataset={
            "path": "mteb/stackexchange-clustering",
            "revision": "6cbc1f7b2bc0622f2e39d2c77fa502909748c259",
        },
        type="Clustering",
        category="s2s",
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
        n_samples={"test": 373850},
        avg_character_length={"test": 57.0},
    )

    def dataset_transform(self):
        ds = dict()
        for split in self.metadata.eval_splits:
            labels = list(itertools.chain.from_iterable(self.dataset[split]["labels"]))
            sentences = list(
                itertools.chain.from_iterable(self.dataset[split]["sentences"])
            )
            ds[split] = Dataset.from_dict({"labels": labels, "sentences": sentences})
        self.dataset = DatasetDict(ds)
        self.dataset = self.stratified_subsampling(
            self.dataset,
            self.seed,
            self.metadata.eval_splits,
            label="labels",
            n_samples=16000,
        )
