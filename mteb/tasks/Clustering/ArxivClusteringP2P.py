from ...abstasks.AbsTaskClustering import AbsTaskClustering


class ArxivClusteringP2P(AbsTaskClustering):
    @property
    def description(self):
        return {
            "name": "ArxivClusteringP2P",
            "hf_hub_name": "mteb/arxiv-clustering-p2p",
            "description": "Clustering of titles+abstract from arxiv. Clustering of 30 sets, either on the main or secondary category",
            "reference": "https://www.kaggle.com/Cornell-University/arxiv",
            "type": "Clustering",
            "category": "p2p",
            "eval_splits": ["test"],
            "eval_langs": ["en"],
            "main_score": "v_measure",
        }
