from ....abstasks import AbsTaskClassification


class ImdbClassification(AbsTaskClassification):
    @property
    def metadata_dict(self):
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
            "revision": "3d86128a09e091d6018b6d26cad27f2739fc2db7",
        }
