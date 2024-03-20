from ....abstasks.AbsTaskClustering import AbsTaskClustering


class RedditClustering(AbsTaskClustering):
    metadata = TaskMetadata()

    @property
    def metadata_dict(self) -> dict[str, str]:
        return dict(self.metadata)
        return {
            "name": "RedditClustering",
            "hf_hub_name": "mteb/reddit-clustering",
            "description": (
                "Clustering of titles from 199 subreddits. Clustering of 25 sets, each with 10-50 classes, and each"
                " class with 100 - 1000 sentences."
            ),
            "reference": "https://arxiv.org/abs/2104.07081",
            "type": "Clustering",
            "category": "s2s",
            "eval_splits": ["test"],
            "eval_langs": ["en"],
            "main_score": "v_measure",
            "revision": "24640382cdbf8abc73003fb0fa6d111a705499eb",
        }
