from mteb.abstasks.clustering_legacy import AbsTaskClusteringLegacy
from mteb.abstasks.task_metadata import TaskMetadata


class HUMEArxivClusteringP2P(AbsTaskClusteringLegacy):
    metadata = TaskMetadata(
        name="HUMEArxivClusteringP2P",
        description="Human evaluation subset of Clustering of titles+abstract from arxiv. Clustering of 30 sets, either on the main or secondary category",
        reference="https://www.kaggle.com/Cornell-University/arxiv",
        dataset={
            "path": "mteb/mteb-human-arxiv-clustering",
            "revision": "6d2f0e9d4f4a51cb54332acaef10478928f0fed8",
        },
        type="Clustering",
        category="t2t",
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
        adapted_from=["ArxivClusteringP2P"],
    )
