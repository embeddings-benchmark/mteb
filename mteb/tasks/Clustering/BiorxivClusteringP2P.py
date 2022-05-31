from ...abstasks.AbsTaskClustering import AbsTaskClustering


class BiorxivClusteringP2P(AbsTaskClustering):
    @property
    def description(self):
        return {
            "name": "BiorxivClusteringP2P",
            "hf_hub_name": "mteb/biorxiv-clustering-p2p",
            "description": "Clustering of titles+abstract from biorxiv. Clustering of 10 sets, based on the main category.",
            "reference": "https://api.biorxiv.org/",
            "type": "Clustering",
            "category": "p2p",
            "eval_splits": ["test"],
            "eval_langs": ["en"],
            "main_score": "v_measure",
        }
