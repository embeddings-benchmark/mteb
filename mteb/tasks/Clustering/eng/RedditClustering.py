from __future__ import annotations

import itertools

from datasets import Dataset, DatasetDict
from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskClustering import AbsTaskClustering
from ....abstasks.AbsTaskClusteringFast import AbsTaskClusteringFast


class RedditFastClusteringS2S(AbsTaskClusteringFast):
    metadata = TaskMetadata(
        name="RedditFastClusteringS2S",
        description="Clustering of titles from 199 subreddits. Clustering of 25 sets, each with 10-50 classes, and each class with 100 - 1000 sentences.",
        reference="https://arxiv.org/abs/2104.07081",
        dataset={
            "path": "mteb/reddit-clustering",
            "revision": "24640382cdbf8abc73003fb0fa6d111a705499eb",
        },
        type="Clustering",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="v_measure",
        date=("2021-01-01", "2021-04-14"),
        form=["written"],
        domains=["Web", "Social"],
        task_subtypes=["Thematic clustering"],
        license="Not specified",  # derived from pushshift
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
        n_samples={"test": 16000},
        avg_character_length={"test": 64.7},
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


class RedditClustering(AbsTaskClustering):
    superseeded_by = "RedditFastClusteringS2S"
    metadata = TaskMetadata(
        name="RedditClustering",
        description="Clustering of titles from 199 subreddits. Clustering of 25 sets, each with 10-50 classes, and each class with 100 - 1000 sentences.",
        reference="https://arxiv.org/abs/2104.07081",
        dataset={
            "path": "mteb/reddit-clustering",
            "revision": "24640382cdbf8abc73003fb0fa6d111a705499eb",
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
        n_samples={"test": 420464},
        avg_character_length={"test": 64.7},
    )
