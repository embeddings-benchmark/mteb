from __future__ import annotations

from mteb.abstasks.AbsTaskClustering import AbsTaskClustering
from mteb.abstasks.AbsTaskClusteringFast import clustering_downsample
from mteb.abstasks.TaskMetadata import TaskMetadata


class ArxivClusteringP2P(AbsTaskClustering):
    superseded_by = "ArXivHierarchicalClusteringP2P"

    metadata = TaskMetadata(
        name="ArxivClusteringP2P",
        description="Clustering of titles+abstract from arxiv. Clustering of 30 sets, either on the main or secondary category",
        reference="https://www.kaggle.com/Cornell-University/arxiv",
        dataset={
            "path": "mteb/arxiv-clustering-p2p",
            "revision": "a122ad7f3f0291bf49cc6f4d32aa80929df69d5d",
        },
        type="Clustering",
        category="p2p",
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
    )


class ArxivClusteringP2PFast(AbsTaskClustering):
    superseded_by = "ArXivHierarchicalClusteringP2P"
    # a faster version of the dataset, since it does not sample from the same distribution we can't use the AbsTaskClusteringFast, instead we
    # simply downsample each cluster.

    metadata = TaskMetadata(
        name="ArxivClusteringP2P.v2",
        description="Clustering of titles+abstract from arxiv. Clustering of 30 sets, either on the main or secondary category",
        reference="https://www.kaggle.com/Cornell-University/arxiv",
        dataset={
            "path": "mteb/arxiv-clustering-p2p",
            "revision": "a122ad7f3f0291bf49cc6f4d32aa80929df69d5d",
        },
        type="Clustering",
        category="p2p",
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
    )

    def dataset_transform(self):
        ds = clustering_downsample(self.dataset, self.seed)
        self.dataset = ds
