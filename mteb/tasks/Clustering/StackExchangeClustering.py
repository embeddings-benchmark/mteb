from ...abstasks.AbsTaskClustering import AbsTaskClustering

class StackExchangeClustering(AbsTaskClustering):
    @property
    def description(self):
        return {
            "name": "StackExchangeClustering",
            "description": "Clustering of titles from 121 stackexchanges. Clustering of 25 sets, each with 10-50 classes, and each class with 100 - 1000 sentences.",
            "reference": "https://arxiv.org/abs/2104.07081",
            "type": "clustering",
            "category": "sts",
            "available_splits": ["dev", "test"],
            "main_score": "v_measure",
        }