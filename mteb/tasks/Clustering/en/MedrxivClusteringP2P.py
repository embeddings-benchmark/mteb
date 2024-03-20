from ....abstasks.AbsTaskClustering import AbsTaskClustering


class MedrxivClusteringP2P(AbsTaskClustering):
    metadata = TaskMetadata()

    @property
    def metadata_dict(self) -> dict[str, str]:
        return dict(self.metadata)
        return {
            "name": "MedrxivClusteringP2P",
            "hf_hub_name": "mteb/medrxiv-clustering-p2p",
            "description": (
                "Clustering of titles+abstract from medrxiv. Clustering of 10 sets, based on the main category."
            ),
            "reference": "https://api.biorxiv.org/",
            "type": "Clustering",
            "category": "p2p",
            "eval_splits": ["test"],
            "eval_langs": ["en"],
            "main_score": "v_measure",
            "revision": "e7a26af6f3ae46b30dde8737f02c07b1505bcc73",
        }
