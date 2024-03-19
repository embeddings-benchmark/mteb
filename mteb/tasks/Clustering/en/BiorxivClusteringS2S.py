from mteb.abstasks.AbsTaskClustering import AbsTaskClustering


class BiorxivClusteringS2S(AbsTaskClustering):
    @property
    def metadata_dict(self):
        return {
            "name": "BiorxivClusteringS2S",
            "hf_hub_name": "mteb/biorxiv-clustering-s2s",
            "description": "Clustering of titles from biorxiv. Clustering of 10 sets, based on the main category.",
            "reference": "https://api.biorxiv.org/",
            "type": "Clustering",
            "category": "s2s",
            "eval_splits": ["test"],
            "eval_langs": ["en"],
            "main_score": "v_measure",
            "revision": "258694dd0231531bc1fd9de6ceb52a0853c6d908",
        }
