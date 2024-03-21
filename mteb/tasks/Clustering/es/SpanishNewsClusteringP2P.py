from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskClustering import AbsTaskClustering


class SpanishNewsClusteringP2P(AbsTaskClustering):
    metadata = TaskMetadata(
        name="SpanishNewsClusteringP2P",
        description="Clustering of news articles, 7 topics in total.",
        reference="https://www.kaggle.com/datasets/kevinmorgado/spanish-news-classification",
        hf_hub_name="mteb/spanish_news_clustering",
        type="Clustering",
        category="p2p",
        eval_splits=["test"],
        eval_langs=["es"],
        main_score="v_measure",
        revision="b5edc3d3d7c12c7b9f883e9da50f6732f3624142",
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
        n_samples={},
        avg_character_length={},
    )
