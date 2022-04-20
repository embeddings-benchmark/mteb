from ...abstasks.AbsTaskSTS import AbsTaskSTS

class STS15STS(AbsTaskSTS):
    @property
    def description(self):
        return {
            "name": "STS15",
            "hf_hub_name": "mteb/sts15-sts",
            "description": "SemEval STS 2015 dataset",
            "reference": "http://alt.qcri.org/semeval2015/task2/",
            "type": "STS",
            "category": "s2s",
            "available_splits": ["test"],
            "main_score": "cosine_spearman",
            "min_score": 0,
            "max_score": 5,
        }