from mteb.abstasks.AbsTaskSTS import AbsTaskSTS


class SickrPLSTS(AbsTaskSTS):
    metadata = 

@property
    def metadata_dict(self) -> dict[str, str]:
        return dict(self.metadata)
        return {
            "name": "SICK-R-PL",
            "hf_hub_name": "PL-MTEB/sickr-pl-sts",
            "description": "Polish version of SICK dataset for textual relatedness.",
            "reference": "https://aclanthology.org/2020.lrec-1.207.pdf",
            "type": "STS",
            "category": "s2s",
            "eval_splits": ["test"],
            "eval_langs": ["pl"],
            "main_score": "cosine_spearman",
            "min_score": 1,
            "max_score": 5,
        }


class CdscrSTS(AbsTaskSTS):
    metadata = 

@property
    def metadata_dict(self) -> dict[str, str]:
        return dict(self.metadata)
        return {
            "name": "CDSC-R",
            "hf_hub_name": "PL-MTEB/cdscr-sts",
            "description": "Compositional Distributional Semantics Corpus for textual relatedness.",
            "reference": "https://aclanthology.org/P17-1073.pdf",
            "type": "STS",
            "category": "s2s",
            "eval_splits": ["test"],
            "eval_langs": ["pl"],
            "main_score": "cosine_spearman",
            "min_score": 1,
            "max_score": 5,
        }
