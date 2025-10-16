import random

from datasets import Dataset, DatasetDict

from mteb.abstasks.clustering_legacy import AbsTaskClusteringLegacy
from mteb.abstasks.task_metadata import TaskMetadata


class ArxivClusteringP2P(AbsTaskClusteringLegacy):
    metadata = TaskMetadata(
        name="ArxivClusteringP2P",
        description="Clustering of titles+abstract from arxiv. Clustering of 30 sets, either on the main or secondary category",
        reference="https://www.kaggle.com/Cornell-University/arxiv",
        dataset={
            "path": "mteb/arxiv-clustering-p2p",
            "revision": "a122ad7f3f0291bf49cc6f4d32aa80929df69d5d",
        },
        type="Clustering",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="v_measure",
        date=("1991-01-01", "2021-01-01"),  # 1991-01-01 is the first arxiv paper
        domains=["Academic", "Written"],
        task_subtypes=[],
        license="cc0-1.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@misc{arxiv_org_submitters_2024,
  author = {arXiv.org submitters},
  doi = {10.34740/KAGGLE/DSV/7548853},
  publisher = {Kaggle},
  title = {arXiv Dataset},
  url = {https://www.kaggle.com/dsv/7548853},
  year = {2024},
}
""",
        prompt="Identify the main and secondary category of Arxiv papers based on the titles and abstracts",
        superseded_by="ArXivHierarchicalClusteringP2P",
    )


class ArxivClusteringP2PFast(AbsTaskClusteringLegacy):
    metadata = TaskMetadata(
        name="ArxivClusteringP2P.v2",
        description="Clustering of titles+abstract from arxiv. Clustering of 30 sets, either on the main or secondary category",
        reference="https://www.kaggle.com/Cornell-University/arxiv",
        dataset={
            "path": "mteb/arxiv-clustering-p2p",
            "revision": "a122ad7f3f0291bf49cc6f4d32aa80929df69d5d",
        },
        type="Clustering",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="v_measure",
        date=("1991-01-01", "2021-01-01"),  # 1991-01-01 is the first arxiv paper
        domains=["Academic", "Written"],
        task_subtypes=[],
        license="cc0-1.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@misc{arxiv_org_submitters_2024,
  author = {arXiv.org submitters},
  doi = {10.34740/KAGGLE/DSV/7548853},
  publisher = {Kaggle},
  title = {arXiv Dataset},
  url = {https://www.kaggle.com/dsv/7548853},
  year = {2024},
}
""",  # None found
        prompt="Identify the main and secondary category of Arxiv papers based on the titles and abstracts",
        adapted_from=["ArxivClusteringP2P"],
        superseded_by="ArXivHierarchicalClusteringP2P",
        # a faster version of the dataset, since it does not sample from the same distribution we can't use the AbsTaskClustering, instead we
        # simply downsample each cluster.
    )

    def dataset_transform(self):
        rng_state = random.Random(self.seed)

        ds = {}
        for split in self.dataset:
            _docs = []
            _labels = []

            n_clusters = len(self.dataset[split])

            for i in range(n_clusters):
                labels = self.dataset[split]["labels"][i]
                sentences = self.dataset[split]["sentences"][i]

                n_sample = min(2048, len(sentences))

                # sample n_sample from each cluster
                idxs = rng_state.sample(range(len(sentences)), n_sample)
                _docs.append([sentences[idx] for idx in idxs])
                _labels.append([labels[idx] for idx in idxs])

            ds[split] = Dataset.from_dict({"sentences": _docs, "labels": _labels})
        self.dataset = DatasetDict(ds)
