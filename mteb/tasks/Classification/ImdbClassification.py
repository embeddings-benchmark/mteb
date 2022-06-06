from ...abstasks import AbsTaskClassification


class ImdbClassification(AbsTaskClassification):
    @property
    def description(self):
        return {
            "name": "ImdbClassification",
            "hf_hub_name": "mteb/imdb",
            "description": "Large Movie Review Dataset",
            "reference": "http://www.aclweb.org/anthology/P11-1015",
            "category": "p2p",
            "type": "Classification",
            "eval_splits": ["test"],
            "eval_langs": ["en"],
            "main_score": "accuracy",
        }
