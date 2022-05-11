from ...abstasks.AbsTaskClustering import AbsTaskClustering

class MedrxivClusteringS2S(AbsTaskClustering):
    @property
    def description(self):
        return {
            "name": "MedrxivClusteringS2S",
            "hf_hub_name": "mteb/medrxiv-clustering-s2s",
            "description": "Clustering of titles from medrxiv. Clustering of 10 sets, based on the main category.",
            "reference": "https://api.biorxiv.org/",
            "type": "Clustering",
            "category": "s2s",
            "available_splits": ["test"],
            "main_score": "v_measure",
        }