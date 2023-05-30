from ...abstasks.AbsTaskClustering import AbsTaskClustering


class TenKGnadClusteringP2P(AbsTaskClustering):
    @property
    def description(self):
        return {
            "name": "TenKGnadClusteringP2P",
            "hf_hub_name": "slvnwhrl/tenkgnad-clustering-p2p",
            "description": "Clustering of news article titles+subheadings+texts. Clustering of 10 splits on the news article category.",
            "reference": "https://github.com/tblock/10kGNAD",
            "type": "Clustering",
            "category": "s2s",
            "eval_splits": ["test"],
            "eval_langs": ["de"],
            "main_score": "v_measure",
            "revision": "a2dd5b02a77de3466a3eaa98ae586b5610314496",
        }
