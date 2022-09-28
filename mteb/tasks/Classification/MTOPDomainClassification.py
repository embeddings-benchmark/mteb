from ...abstasks import AbsTaskClassification, MultilingualTask


_LANGUAGES = ["en", "de", "es", "fr", "hi", "th"]


class MTOPDomainClassification(MultilingualTask, AbsTaskClassification):
    @property
    def description(self):
        return {
            "name": "MTOPDomainClassification",
            "hf_hub_name": "mteb/mtop_domain",
            "description": "MTOP: Multilingual Task-Oriented Semantic Parsing",
            "reference": "https://arxiv.org/pdf/2008.09335.pdf",
            "category": "s2s",
            "type": "Classification",
            "eval_splits": ["validation", "test"],
            "eval_langs": _LANGUAGES,
            "main_score": "accuracy",
            "revision": "d80d48c1eb48d3562165c59d59d0034df9fff0bf",
        }
