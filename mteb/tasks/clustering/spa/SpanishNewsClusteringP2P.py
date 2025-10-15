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
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation=None,
    )
