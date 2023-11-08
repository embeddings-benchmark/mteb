from ...abstasks.AbsTaskClustering import AbsTaskClustering


class BiorxivClusteringS2S_Spanish(AbsTaskClustering):
    @property
    def description(self):
        return {
            "name": "BiorxivClusteringS2S_Spanish",
            "hf_hub_name": "clibrain/biorxiv-clustering-es",
            "description": "Clustering of titles from biorxiv. Clustering of 10 sets, based on the main category.",
            "reference": "https://api.biorxiv.org/",
            "type": "Clustering",
            "category": "s2s",
            "eval_splits": ["test"],
            "eval_langs": ["es"],
            "main_score": "v_measure",
        }
