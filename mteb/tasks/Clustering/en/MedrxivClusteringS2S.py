from ....abstasks.AbsTaskClustering import AbsTaskClustering


class MedrxivClusteringS2S(AbsTaskClustering):
    @property
    def metadata_dict(self):
        return {
            "name": "MedrxivClusteringS2S",
            "hf_hub_name": "mteb/medrxiv-clustering-s2s",
            "description": "Clustering of titles from medrxiv. Clustering of 10 sets, based on the main category.",
            "reference": "https://api.biorxiv.org/",
            "type": "Clustering",
            "category": "s2s",
            "eval_splits": ["test"],
            "eval_langs": ["en"],
            "main_score": "v_measure",
            "revision": "35191c8c0dca72d8ff3efcd72aa802307d469663",
        }
