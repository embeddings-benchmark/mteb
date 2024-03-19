from ....abstasks.AbsTaskClustering import AbsTaskClustering


class RedditClusteringP2P(AbsTaskClustering):
    @property
    def metadata_dict(self):
        return {
            "name": "RedditClusteringP2P",
            "hf_hub_name": "mteb/reddit-clustering-p2p",
            "description": (
                "Clustering of title+posts from reddit. Clustering of 10 sets with 1K - 100K samples and 10 - 100 labels each."
            ),
            "reference": "https://huggingface.co/datasets/sentence-transformers/reddit-title-body",
            "type": "Clustering",
            "category": "p2p",
            "eval_splits": ["test"],
            "eval_langs": ["en"],
            "main_score": "v_measure",
            "revision": "282350215ef01743dc01b456c7f5241fa8937f16",
        }
