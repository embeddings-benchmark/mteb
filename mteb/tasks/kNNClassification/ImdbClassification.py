from ...abstasks import AbsTaskKNNClassification


class ImdbClassification(AbsTaskKNNClassification):
    @property
    def description(self):
        return {
            "name": "ImdbClassification",
            "hf_hub_name": "mteb/imdb",
            "description": "Large Movie Review Dataset",
            "reference": "http://www.aclweb.org/anthology/P11-1015",
            "category": "p2p",
            "type": "kNNClassification",
            "available_splits": ["train", "test"],
            "available_langs": ["en"],
            "main_score": "accuracy",
        }
