from ...abstasks import AbsTaskClassification


class PatentClassification(AbsTaskClassification):
    @property
    def description(self):
        return {
            "name": "PatentClassification",
            "hf_hub_name": "jinaai/small_patent",
            "description": "Patent classification evaluation based on the test set of the big patent dataset",
            "reference": "https://dblp.org/rec/journals/corr/abs-1906-03741.bib",
            "category": "p2p",
            "type": "Classification",
            "eval_splits": ["test"],
            "eval_langs": ["en"],
            "main_score": "accuracy",
            "revision": "01ab216c0a33cb4163a3ac0f69b9e573017382fd",
        }
