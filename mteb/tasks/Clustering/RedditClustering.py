from ...abstasks.AbsTaskClustering import AbsTaskClustering

class RedditClustering(AbsTaskClustering):
    @property
    def description(self):
        return {
            "name": "RedditClustering",
            "hf_hub_name": "mteb/reddit-clustering",
            "description": "Clustering of titles from 199 subreddits. Clustering of 25 sets, each with 10-50 classes, and each class with 100 - 1000 sentences.",
            "reference": "https://arxiv.org/abs/2104.07081",
            "type": "Clustering",
            "category": "sts",
            "available_splits": ["test", "validation"],
            "main_score": "v_measure",
        }