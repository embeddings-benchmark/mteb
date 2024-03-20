from ....abstasks.AbsTaskSTS import AbsTaskSTS


class STS16STS(AbsTaskSTS):
    metadata = 

@property
    def metadata_dict(self) -> dict[str, str]:
        return dict(self.metadata)
        return {
            "name": "STS16",
            "hf_hub_name": "mteb/sts16-sts",
            "description": "SemEval STS 2016 dataset",
            "reference": "http://alt.qcri.org/semeval2016/task1/",
            "type": "STS",
            "category": "s2s",
            "eval_splits": ["test"],
            "eval_langs": ["en"],
            "main_score": "cosine_spearman",
            "min_score": 0,
            "max_score": 5,
            "revision": "4d8694f8f0e0100860b497b999b3dbed754a0513",
        }
