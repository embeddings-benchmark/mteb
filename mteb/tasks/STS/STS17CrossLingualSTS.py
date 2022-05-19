from ...abstasks.AbsTaskSTS import AbsTaskSTS


class STS16STS(AbsTaskSTS):
    @property
    def description(self):
        return {
            "name": "STS17",
            "hf_hub_name": "mteb/sts17-crosslingual-sts",
            "description": "STS 2017dataset",
            "reference": "http://alt.qcri.org/semeval2016/task1/",
            "type": "STS",
            "category": "s2s",
            "available_splits": ["test"],
            "available_langs": ['ar-ar','en-ar','en-de','en-en','en-tr','es-en','es-es','fr-en','it-en','nl-en'],
            "main_score": "cosine_spearman",
            "min_score": 0,
            "max_score": 5,
        }
