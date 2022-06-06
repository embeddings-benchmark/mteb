from ...abstasks import AbsTaskSTS, CrosslingualTask

_LANGUAGES = {
    "en": "en",
    "de": "de",
    "es": "es",
    "pl": "pl",
    "tr": "tr",
    "ar": "ar",
    "ru": "ru",
    "zh": "zh",
    "fr": "fr",
    "de-en": "de-en",
    "es-en": "es-en",
    "it": "it",
    "pl-en": "pl-en",
    "zh-en": "zh-en",
    "es-it": "es-it",
    "de-fr": "de-fr",
    "de-pl": "de-pl",
    "fr-pl": "fr-pl",
}


class STS22CrosslingualSTS(AbsTaskSTS, CrosslingualTask):
    @property
    def description(self):
        return {
            "name": "STS22",
            "hf_hub_name": "mteb/sts22-crosslingual-sts",
            "description": "SemEval 2022 Task 8: Multilingual News Article Similarity",
            "reference": "https://competitions.codalab.org/competitions/33835",
            "type": "STS",
            "category": "s2s",
            "eval_splits": ["test"],
            "eval_langs": _LANGUAGES,
            "main_score": "cosine_spearman",
            "min_score": 0,
            "max_score": 5,
        }
