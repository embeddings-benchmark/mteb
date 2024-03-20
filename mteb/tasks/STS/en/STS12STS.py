from ....abstasks.AbsTaskSTS import AbsTaskSTS


class STS12STS(AbsTaskSTS):
    metadata = 

@property
    def metadata_dict(self) -> dict[str, str]:
        return dict(self.metadata)
        return {
            "name": "STS12",
            "hf_hub_name": "mteb/sts12-sts",
            "description": "SemEval STS 2012 dataset.",
            "reference": "https://www.aclweb.org/anthology/S12-1051.pdf",
            "type": "STS",
            "category": "s2s",
            "eval_splits": ["test"],
            "eval_langs": ["en"],
            "main_score": "cosine_spearman",
            "min_score": 0,
            "max_score": 5,
            "revision": "a0d554a64d88156834ff5ae9920b964011b16384",
        }
