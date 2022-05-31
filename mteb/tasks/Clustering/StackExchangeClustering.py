from ...abstasks.AbsTaskClustering import AbsTaskClustering


class StackExchangeClustering(AbsTaskClustering):
    @property
    def description(self):
        return {
            "name": "StackExchangeClustering",
            "hf_hub_name": "mteb/stackexchange-clustering",
            "description": "Clustering of titles from 121 stackexchanges. Clustering of 25 sets, each with 10-50 classes, and each class with 100 - 1000 sentences.",
            "reference": "https://arxiv.org/abs/2104.07081",
            "type": "Clustering",
            "category": "s2s",
            "available_splits": ["validation", "test"],
            "eval_langs": ["en"],
            "main_score": "v_measure",
        }
