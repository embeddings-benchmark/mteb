from __future__ import annotations

import itertools

import numpy as np
from datasets import Dataset, DatasetDict

from mteb.abstasks.AbsTaskClustering import AbsTaskClustering
from mteb.abstasks.AbsTaskClusteringFast import (
    AbsTaskClusteringFast,
    check_label_distribution,
)
from mteb.abstasks.TaskMetadata import TaskMetadata

NUM_SAMPLES = 2048


class BlurbsClusteringS2S(AbsTaskClustering):
    superseded_by = "BlurbsClusteringS2S.v2"

    metadata = TaskMetadata(
        name="BlurbsClusteringS2S",
        description="Clustering of book titles. Clustering of 28 sets, either on the main or secondary genre.",
        reference="https://www.inf.uni-hamburg.de/en/inst/ab/lt/resources/data/germeval-2019-hmc.html",
        dataset={
            "path": "slvnwhrl/blurbs-clustering-s2s",
            "revision": "22793b6a6465bf00120ad525e38c51210858132c",
        },
        type="Clustering",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["deu-Latn"],
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
        bibtex_citation="""@inproceedings{Remus2019GermEval2T,
  title={GermEval 2019 Task 1: Hierarchical Classification of Blurbs},
  author={Steffen Remus and Rami Aly and Chris Biemann},
  booktitle={Conference on Natural Language Processing},
  year={2019},
  url={https://api.semanticscholar.org/CorpusID:208334484}
}""",
        n_samples={"test": 174637},
        avg_character_length={"test": 23.02},
    )


class BlurbsClusteringS2SFast(AbsTaskClusteringFast):
    # a faster version of the task, since it does not sample from the same distribution we can't use the AbsTaskClusteringFast, instead we
    # simply downsample each cluster.

    max_document_to_embed = NUM_SAMPLES
    max_fraction_of_documents_to_embed = None

    metadata = TaskMetadata(
        name="BlurbsClusteringS2S.v2",
        description="Clustering of book titles. Clustering of 28 sets, either on the main or secondary genre.",
        reference="https://www.inf.uni-hamburg.de/en/inst/ab/lt/resources/data/germeval-2019-hmc.html",
        dataset={
            "path": "slvnwhrl/blurbs-clustering-s2s",
            "revision": "22793b6a6465bf00120ad525e38c51210858132c",
        },
        type="Clustering",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["deu-Latn"],
        main_score="v_measure",
        date=(
            "1900-01-01",
            "2019-12-31",
        ),  # since it is books it is likely to be from the 20th century -> paper from 2019
        form=["written"],
        domains=["Fiction"],
        task_subtypes=["Thematic clustering"],
        license="cc-by-nc-4.0",
        socioeconomic_status="mixed",
        annotations_creators="derived",
        dialect=[],
        text_creation="found",
        bibtex_citation="""@inproceedings{Remus2019GermEval2T,
  title={GermEval 2019 Task 1: Hierarchical Classification of Blurbs},
  author={Steffen Remus and Rami Aly and Chris Biemann},
  booktitle={Conference on Natural Language Processing},
  year={2019},
  url={https://api.semanticscholar.org/CorpusID:208334484}
}""",
        n_samples={"test": NUM_SAMPLES},
        avg_character_length={"test": 23.02},
    )

    def dataset_transform(self):
        ds = dict()
        for split in self.metadata.eval_splits:
            labels = list(itertools.chain.from_iterable(self.dataset[split]["labels"]))
            sentences = list(
                itertools.chain.from_iterable(self.dataset[split]["sentences"])
            )

            check_label_distribution(self.dataset[split])

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
            n_samples=NUM_SAMPLES,
        )
