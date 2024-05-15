from __future__ import annotations

import itertools

from datasets import Dataset, DatasetDict
import numpy as np

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskClustering import AbsTaskClustering
from ....abstasks.AbsTaskClusteringFast import AbsTaskClusteringFast


class RedditClusteringP2P(AbsTaskClustering):
    metadata = TaskMetadata(
        name="RedditClusteringP2P",
        description="Clustering of title+posts from reddit. Clustering of 10 sets of 50k paragraphs and 40 sets of 10k paragraphs.",
        reference="https://arxiv.org/abs/2104.07081",
        dataset={
            "path": "mteb/reddit-clustering-p2p",
            "revision": "385e3cb46b4cfa89021f56c4380204149d0efe33",
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
        n_samples={"test": 459399},
        avg_character_length={"test": 727.7},
    )


class RedditFastClusteringP2P(AbsTaskClusteringFast):
    metadata = TaskMetadata(
        name="RedditFastClusteringP2P",
        description="Clustering of title+posts from reddit. Clustering of 10 sets of 50k paragraphs and 40 sets of 10k paragraphs.",
        reference="https://arxiv.org/abs/2104.07081",
        dataset={
            "path": "mteb/reddit-clustering-p2p",
            "revision": "385e3cb46b4cfa89021f56c4380204149d0efe33",
        },
        type="Clustering",
        category="p2p",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="v_measure",
        date=("2021-01-01", "2021-04-14"),
        form=["written"],
        domains=["Web", "Social"],
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
        n_samples={"test": 16000},
        avg_character_length={"test": 727.7},
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
            solo_label_idx = np.where(counts==1)
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
