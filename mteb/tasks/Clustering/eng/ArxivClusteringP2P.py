from __future__ import annotations

from mteb.abstasks.AbsTaskClustering import AbsTaskClustering
from mteb.abstasks.AbsTaskClusteringFast import clustering_downsample
from mteb.abstasks.TaskMetadata import TaskMetadata


class ArxivClusteringP2P(AbsTaskClustering):
    superseeded_by = "ArXivHierarchicalClusteringP2P"

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
        bibtex_citation="""@misc{arxiv_org_submitters_2024,
    title={arXiv Dataset},
    url={https://www.kaggle.com/dsv/7548853},
    DOI={10.34740/KAGGLE/DSV/7548853},
    publisher={Kaggle},
    author={arXiv.org submitters},
    year={2024}
}""",
        n_samples={"test": 732723},
        avg_character_length={"test": 1009.98},
    )


class ArxivClusteringP2PFast(AbsTaskClustering):
    superseeded_by = "ArXivHierarchicalClusteringP2P"
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
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="v_measure",
        date=("1991-01-01", "2021-01-01"),  # 1991-01-01 is the first arxiv paper
        form=["written"],
        domains=["Academic"],
        task_subtypes=[],
        license="CC0",
        socioeconomic_status="high",
        annotations_creators="derived",
        dialect=[],
        text_creation="found",
        bibtex_citation="""@misc{arxiv_org_submitters_2024,
    title={arXiv Dataset},
    url={https://www.kaggle.com/dsv/7548853},
    DOI={10.34740/KAGGLE/DSV/7548853},
    publisher={Kaggle},
    author={arXiv.org submitters},
    year={2024}
}""",  # None found
        n_samples={"test": 250_000},
        avg_character_length={"test": 1009.98},
    )

    def dataset_transform(self):
        ds = clustering_downsample(self.dataset, self.seed)
        self.dataset = ds
