from ...abstasks.AbsTaskSTS import AbsTaskSTS


class STS14STS(AbsTaskSTS):
    @property
    def description(self):
        return {
            "name": "STS14",
            "hf_hub_name": "mteb/sts14-sts",
            "description": "SemEval STS 2014 dataset. Currently only the English dataset",
            "reference": "http://alt.qcri.org/semeval2014/task10/",
            "type": "STS",
            "category": "s2s",
            "eval_splits": ["test"],
            "eval_langs": ["en"],
            "main_score": "cosine_spearman",
            "min_score": 0,
            "max_score": 5,
            "revision": "6031580fec1f6af667f0bd2da0a551cf4f0b2375",
        }
