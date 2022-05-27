from ...abstasks import AbsTaskClassification, MultilingualTask

_LANGUAGES = ["en", "de", "es", "fr", "hi", "th"]


class MTOPIntentClassification(MultilingualTask, AbsTaskClassification):
    @property
    def description(self):
        return {
            "name": "MTOPIntentClassification",
            "hf_hub_name": "mteb/mtop_intent",
            "description": "MTOP: Multilingual Task-Oriented Semantic Parsing",
            "reference": "https://arxiv.org/pdf/2008.09335.pdf",
            "category": "s2s",
            "type": "Classification",
            "available_splits": ["train", "validation", "test"],
            "available_langs": _LANGUAGES,
            "main_score": "accuracy",
        }
