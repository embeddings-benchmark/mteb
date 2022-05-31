from ...abstasks.AbsTaskClustering import AbsTaskClustering


class ArxivClusteringS2S(AbsTaskClustering):
    @property
    def description(self):
        return {
            "name": "ArxivClusteringS2S",
            "hf_hub_name": "mteb/arxiv-clustering-s2s",
            "description": "Clustering of titles from arxiv. Clustering of 30 sets, either on the main or secondary category",
            "reference": "https://www.kaggle.com/Cornell-University/arxiv",
            "type": "Clustering",
            "category": "s2s",
            "available_splits": ["test"],
            "eval_langs": ["en"],
            "main_score": "v_measure",
        }
