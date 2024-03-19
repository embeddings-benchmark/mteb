from ....abstasks.AbsTaskClustering import AbsTaskClustering


class TenKGnadClusteringP2P(AbsTaskClustering):
    @property
    def metadata_dict(self):
        return {
            "name": "TenKGnadClusteringP2P",
            "hf_hub_name": "slvnwhrl/tenkgnad-clustering-p2p",
            "description": "Clustering of news article titles+subheadings+texts. Clustering of 10 splits on the news article category.",
            "reference": "https://tblock.github.io/10kGNAD/",
            "type": "Clustering",
            "category": "p2p",
            "eval_splits": ["test"],
            "eval_langs": ["de"],
            "main_score": "v_measure",
            "revision": "5c59e41555244b7e45c9a6be2d720ab4bafae558",
        }
