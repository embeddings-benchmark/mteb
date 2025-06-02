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
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="v_measure",
        date=("2021-01-01", "2021-04-14"),
        domains=["Web", "Written"],
        task_subtypes=["Thematic clustering"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@article{geigle:2021:arxiv,
  archiveprefix = {arXiv},
  author = {Gregor Geigle and
Nils Reimers and
Andreas R{\"u}ckl{\'e} and
Iryna Gurevych},
  eprint = {2104.07081},
  journal = {arXiv preprint},
  title = {TWEAC: Transformer with Extendable QA Agent Classifiers},
  url = {http://arxiv.org/abs/2104.07081},
  volume = {abs/2104.07081},
  year = {2021},
}
""",
        prompt="Identify the topic or theme of StackExchange posts based on the titles",
        adapted_from=["StackExchangeClustering"],
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
        self.dataset = self.stratified_subsampling(
            self.dataset,
            self.seed,
            self.metadata.eval_splits,
            label="labels",
        )
        self.max_fraction_of_documents_to_embed = None


class StackExchangeClustering(AbsTaskClustering):
    superseded_by = "StackExchangeClustering.v2"
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
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="v_measure",
        date=("2021-01-01", "2021-04-14"),
        domains=["Web", "Written"],
        task_subtypes=["Thematic clustering"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@article{geigle:2021:arxiv,
  archiveprefix = {arXiv},
  author = {Gregor Geigle and
Nils Reimers and
Andreas R{\"u}ckl{\'e} and
Iryna Gurevych},
  eprint = {2104.07081},
  journal = {arXiv preprint},
  title = {TWEAC: Transformer with Extendable QA Agent Classifiers},
  url = {http://arxiv.org/abs/2104.07081},
  volume = {abs/2104.07081},
  year = {2021},
}
""",
        prompt="Identify the topic or theme of StackExchange posts based on the titles",
    )
