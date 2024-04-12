from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskClustering import AbsTaskClustering


class SpanishNewsClusteringP2P(AbsTaskClustering):
    metadata = TaskMetadata(
        name="SpanishNewsClusteringP2P",
        description="Clustering of news articles, 7 topics in total.",
        reference="https://www.kaggle.com/datasets/kevinmorgado/spanish-news-classification",
        dataset={
            "path": "jinaai/spanish_news_clustering",
            "revision": "bf8ca8ddc5b7da4f7004720ddf99bbe0483480e6",
        },
        type="Clustering",
        category="p2p",
        eval_splits=["test"],
        eval_langs=["spa-Latn"],
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
        n_samples=None,
        avg_character_length=None,
    )
