from ...abstasks.AbsTaskSTS import AbsTaskSTS


class STS12STS(AbsTaskSTS):
    @property
    def description(self):
        return {
            "name": "STS12",
            "hf_hub_name": "mteb/sts12-sts",
            "description": "SemEval STS 2012 dataset.",
            "reference": "https://www.aclweb.org/anthology/S12-1051.pdf",
            "type": "STS",
            "category": "s2s",
            "available_splits": ["train", "test"],
            "available_langs": ["en"],
            "main_score": "cosine_spearman",
            "min_score": 0,
            "max_score": 5,
        }
