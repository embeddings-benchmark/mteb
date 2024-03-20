from ....abstasks.AbsTaskClustering import AbsTaskClustering


class StackExchangeClustering(AbsTaskClustering):
    metadata = TaskMetadata()

    @property
    def metadata_dict(self) -> dict[str, str]:
        return dict(self.metadata)
        return {
            "name": "StackExchangeClustering",
            "hf_hub_name": "mteb/stackexchange-clustering",
            "description": (
                "Clustering of titles from 121 stackexchanges. Clustering of 25 sets, each with 10-50 classes, and"
                " each class with 100 - 1000 sentences."
            ),
            "reference": "https://arxiv.org/abs/2104.07081",
            "type": "Clustering",
            "category": "s2s",
            "eval_splits": ["validation", "test"],
            "eval_langs": ["en"],
            "main_score": "v_measure",
            "revision": "6cbc1f7b2bc0622f2e39d2c77fa502909748c259",
        }
