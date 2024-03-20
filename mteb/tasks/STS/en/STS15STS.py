from ....abstasks.AbsTaskSTS import AbsTaskSTS


class STS15STS(AbsTaskSTS):
    metadata = TaskMetadata()

    @property
    def metadata_dict(self) -> dict[str, str]:
        return dict(self.metadata)
        return {
            "name": "STS15",
            "hf_hub_name": "mteb/sts15-sts",
            "description": "SemEval STS 2015 dataset",
            "reference": "http://alt.qcri.org/semeval2015/task2/",
            "type": "STS",
            "category": "s2s",
            "eval_splits": ["test"],
            "eval_langs": ["en"],
            "main_score": "cosine_spearman",
            "min_score": 0,
            "max_score": 5,
            "revision": "ae752c7c21bf194d8b67fd573edf7ae58183cbe3",
        }
