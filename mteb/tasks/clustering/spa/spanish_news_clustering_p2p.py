from mteb.abstasks.clustering_legacy import AbsTaskClusteringLegacy
from mteb.abstasks.task_metadata import TaskMetadata


class SpanishNewsClusteringP2P(AbsTaskClusteringLegacy):
    metadata = TaskMetadata(
        name="SpanishNewsClusteringP2P",
        description="Clustering of news articles, 7 topics in total.",
        reference="https://www.kaggle.com/datasets/kevinmorgado/spanish-news-classification",
        dataset={
            "path": "jinaai/spanish_news_clustering",
            "revision": "bf8ca8ddc5b7da4f7004720ddf99bbe0483480e6",
        },
        type="Clustering",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["spa-Latn"],
        main_score="v_measure",
        date=("2017-01-01", "2019-12-31"),
        domains=["News", "Written"],
        task_subtypes=["Thematic clustering"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@misc{kevinmorgado2019spanish,
  author = {Kevin Morgado},
  howpublished = {Kaggle},
  title = {Spanish News Classification},
  url = {https://www.kaggle.com/datasets/kevinmorgado/spanish-news-classification},
  year = {2019},
}
""",
    )
