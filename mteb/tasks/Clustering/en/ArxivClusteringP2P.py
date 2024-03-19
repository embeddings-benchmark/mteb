from ....abstasks.AbsTaskClustering import AbsTaskClustering


class ArxivClusteringP2P(AbsTaskClustering):
    @property
    def metadata_dict(self):
        return {
            "name": "ArxivClusteringP2P",
            "hf_hub_name": "mteb/arxiv-clustering-p2p",
            "description": (
                "Clustering of titles+abstract from arxiv. Clustering of 30 sets, either on the main or secondary"
                " category"
            ),
            "reference": "https://www.kaggle.com/Cornell-University/arxiv",
            "type": "Clustering",
            "category": "p2p",
            "eval_splits": ["test"],
            "eval_langs": ["en"],
            "main_score": "v_measure",
            "revision": "a122ad7f3f0291bf49cc6f4d32aa80929df69d5d",
        }
