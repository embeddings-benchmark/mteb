from ....abstasks.AbsTaskClustering import AbsTaskClustering


class SpanishNewsClusteringP2P(AbsTaskClustering):
    metadata = TaskMetadata()

    @property
    def metadata_dict(self) -> dict[str, str]:
        return dict(self.metadata)
        return {
            "name": "SpanishNewsClusteringP2P",
            "hf_hub_name": "jinaai/spanish_news_clustering",
            "description": ("Clustering of news articles, 7 topics in total."),
            "reference": "https://www.kaggle.com/datasets/kevinmorgado/spanish-news-classification",
            "type": "Clustering",
            "category": "p2p",
            "eval_splits": ["test"],
            "eval_langs": ["es"],
            "main_score": "v_measure",
            "revision": "b5edc3d3d7c12c7b9f883e9da50f6732f3624142",
        }
