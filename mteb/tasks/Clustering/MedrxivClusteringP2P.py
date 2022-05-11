from ...abstasks.AbsTaskClustering import AbsTaskClustering


class MedrxivClusteringP2P(AbsTaskClustering):
    @property
    def description(self):
        return {
            "name": "MedrxivClusteringP2P",
            "hf_hub_name": "mteb/medrxiv-clustering-p2p",
            "description": "Clustering of titles+abstract from medrxiv. Clustering of 10 sets, based on the main category.",
            "reference": "https://api.biorxiv.org/",
            "type": "Clustering",
            "category": "p2p",
            "available_splits": ["test"],
            "main_score": "v_measure",
        }
