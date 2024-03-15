from ....abstasks.AbsTaskClustering import AbsTaskClustering


class TenKGnadClusteringS2S(AbsTaskClustering):
    @property
    def description(self):
        return {
            "name": "TenKGnadClusteringS2S",
            "hf_hub_name": "slvnwhrl/tenkgnad-clustering-s2s",
            "description": "Clustering of news article titles. Clustering of 10 splits on the news article category.",
            "reference": "https://tblock.github.io/10kGNAD/",
            "type": "Clustering",
            "category": "s2s",
            "eval_splits": ["test"],
            "eval_langs": ["de"],
            "main_score": "v_measure",
            "revision": "6cddbe003f12b9b140aec477b583ac4191f01786",
        }
