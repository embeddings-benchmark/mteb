from ...abstasks.AbsTaskSTS import AbsTaskSTS


class STS16STS(AbsTaskSTS):
    @property
    def description(self):
        return {
            "name": "STS16",
            "hf_hub_name": "mteb/sts16-sts",
            "description": "SemEval STS 2016 dataset",
            "reference": "http://alt.qcri.org/semeval2016/task1/",
            "type": "STS",
            "category": "s2s",
            "available_splits": ["test"],
            "main_score": "cosine_spearman",
            "min_score": 0,
            "max_score": 5,
        }
