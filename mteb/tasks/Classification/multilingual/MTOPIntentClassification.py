from ....abstasks import AbsTaskClassification, MultilingualTask

_LANGUAGES = ["en", "de", "es", "fr", "hi", "th"]


class MTOPIntentClassification(MultilingualTask, AbsTaskClassification):
    @property
    def metadata_dict(self):
        return {
            "name": "MTOPIntentClassification",
            "hf_hub_name": "mteb/mtop_intent",
            "description": "MTOP: Multilingual Task-Oriented Semantic Parsing",
            "reference": "https://arxiv.org/pdf/2008.09335.pdf",
            "category": "s2s",
            "type": "Classification",
            "eval_splits": ["validation", "test"],
            "eval_langs": _LANGUAGES,
            "main_score": "accuracy",
            "revision": "ae001d0e6b1228650b7bd1c2c65fb50ad11a8aba",
        }
