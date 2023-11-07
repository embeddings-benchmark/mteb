from ...abstasks.AbsTaskClustering import AbsTaskClustering


class RedditClustering_Spanish(AbsTaskClustering):
    @property
    def description(self):
        return {
            "name": "RedditClusteringSpanish",
            "hf_hub_name": "clibrain/reddit-s2s-spanish",
            "description": (
                "Clustering of titles from 199 subreddits. Clustering of 25 sets, each with 10-50 classes, and each"
                " class with 100 - 1000 sentences."
            ),
            "reference": "https://arxiv.org/abs/2104.07081",
            "type": "Clustering",
            "category": "s2s",
            "eval_splits": ["test"],
            "eval_langs": ["es"],
            "main_score": "v_measure",
            "revision": "24640382cdbf8abc73003fb0fa6d111a705499eb",
        }
