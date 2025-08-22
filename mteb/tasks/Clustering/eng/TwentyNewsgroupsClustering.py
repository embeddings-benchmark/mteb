from __future__ import annotations

import itertools

from datasets import Dataset, DatasetDict

from mteb.abstasks.AbsTaskClustering import AbsTaskClustering
from mteb.abstasks.AbsTaskClusteringFast import (
    AbsTaskClusteringFast,
    check_label_distribution,
)
from mteb.abstasks.TaskMetadata import TaskMetadata


class TwentyNewsgroupsClustering(AbsTaskClustering):
    superseded_by = "TwentyNewsgroupsClustering.v2"
    metadata = TaskMetadata(
        name="TwentyNewsgroupsClustering",
        description="Clustering of the 20 Newsgroups dataset (subject only).",
        reference="https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html",
        dataset={
            "path": "mteb/twentynewsgroups-clustering",
            "revision": "6125ec4e24fa026cec8a478383ee943acfbd5449",
        },
        type="Clustering",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="v_measure",
        date=("1995-01-01", "1995-01-01"),
        domains=["News", "Written"],
        task_subtypes=["Thematic clustering"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@incollection{LANG1995331,
  address = {San Francisco (CA)},
  author = {Ken Lang},
  booktitle = {Machine Learning Proceedings 1995},
  doi = {https://doi.org/10.1016/B978-1-55860-377-6.50048-7},
  editor = {Armand Prieditis and Stuart Russell},
  isbn = {978-1-55860-377-6},
  pages = {331-339},
  publisher = {Morgan Kaufmann},
  title = {NewsWeeder: Learning to Filter Netnews},
  url = {https://www.sciencedirect.com/science/article/pii/B9781558603776500487},
  year = {1995},
}
""",
        prompt="Identify the topic or theme of the given news articles",
    )


class TwentyNewsgroupsClusteringFast(AbsTaskClusteringFast):
    metadata = TaskMetadata(
        name="TwentyNewsgroupsClustering.v2",
        description="Clustering of the 20 Newsgroups dataset (subject only).",
        reference="https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html",
        dataset={
            "path": "mteb/twentynewsgroups-clustering",
            "revision": "6125ec4e24fa026cec8a478383ee943acfbd5449",
        },
        type="Clustering",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="v_measure",
        date=("1995-01-01", "1995-01-01"),
        domains=["News", "Written"],
        task_subtypes=["Thematic clustering"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@incollection{LANG1995331,
  address = {San Francisco (CA)},
  author = {Ken Lang},
  booktitle = {Machine Learning Proceedings 1995},
  doi = {https://doi.org/10.1016/B978-1-55860-377-6.50048-7},
  editor = {Armand Prieditis and Stuart Russell},
  isbn = {978-1-55860-377-6},
  pages = {331-339},
  publisher = {Morgan Kaufmann},
  title = {NewsWeeder: Learning to Filter Netnews},
  url = {https://www.sciencedirect.com/science/article/pii/B9781558603776500487},
  year = {1995},
}
""",
        prompt="Identify the topic or theme of the given news articles",
        adapted_from=["TwentyNewsgroupsClustering"],
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
