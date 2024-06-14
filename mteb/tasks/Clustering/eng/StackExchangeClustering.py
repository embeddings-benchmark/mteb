from __future__ import annotations

import itertools

from datasets import Dataset, DatasetDict

from mteb.abstasks.AbsTaskClustering import AbsTaskClustering
from mteb.abstasks.AbsTaskClusteringFast import (
    AbsTaskClusteringFast,
    check_label_distribution,
)
from mteb.abstasks.TaskMetadata import TaskMetadata


class StackExchangeClusteringFast(AbsTaskClusteringFast):
    metadata = TaskMetadata(
        name="StackExchangeClustering.v2",
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
        date=("2021-01-01", "2021-04-14"),
        form=["written"],
        domains=["Web"],
        task_subtypes=["Thematic clustering"],
        license="Not specified",
        socioeconomic_status="mixed",
        annotations_creators="derived",
        dialect=[],
        text_creation="found",
        bibtex_citation="""@article{geigle:2021:arxiv,
        author    = {Gregor Geigle and 
                        Nils Reimers and 
                        Andreas R{\"u}ckl{\'e} and
                        Iryna Gurevych},
        title     = {TWEAC: Transformer with Extendable QA Agent Classifiers},
        journal   = {arXiv preprint},
        volume    = {abs/2104.07081},
        year      = {2021},
        url       = {http://arxiv.org/abs/2104.07081},
        archivePrefix = {arXiv},
        eprint    = {2104.07081}
        }""",
        n_samples={"test": 32768},
        avg_character_length={"test": 57.0},
    )

    def dataset_transform(self):
        ds = dict()
        for split in self.metadata.eval_splits:
            labels = list(itertools.chain.from_iterable(self.dataset[split]["labels"]))
            sentences = list(
                itertools.chain.from_iterable(self.dataset[split]["sentences"])
            )

            check_label_distribution(self.dataset[split])

            ds[split] = Dataset.from_dict({"labels": labels, "sentences": sentences})
        self.dataset = DatasetDict(ds)
        self.dataset = self.stratified_subsampling(
            self.dataset,
            self.seed,
            self.metadata.eval_splits,
            label="labels",
            n_samples=32768,
        )
        self.use_dataset_as_is = True


class StackExchangeClustering(AbsTaskClustering):
    superseeded_by = "StackExchangeClustering.v2"
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
        bibtex_citation="""@article{geigle:2021:arxiv,
        author    = {Gregor Geigle and 
                        Nils Reimers and 
                        Andreas R{\"u}ckl{\'e} and
                        Iryna Gurevych},
        title     = {TWEAC: Transformer with Extendable QA Agent Classifiers},
        journal   = {arXiv preprint},
        volume    = {abs/2104.07081},
        year      = {2021},
        url       = {http://arxiv.org/abs/2104.07081},
        archivePrefix = {arXiv},
        eprint    = {2104.07081}
        }""",
        n_samples={"test": 373850},
        avg_character_length={"test": 57.0},
    )
