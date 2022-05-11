from ...abstasks.AbsTaskSTS import AbsTaskSTS


class BiossesSTS(AbsTaskSTS):
    @property
    def description(self):
        return {
            "name": "BIOSSES",
            "hf_hub_name": "mteb/biosses-sts",
            "description": "Biomedical Semantic Similarity Estimation.",
            "reference": "https://tabilab.cmpe.boun.edu.tr/BIOSSES/DataSet.html",
            "type": "STS",
            "category": "s2s",
            "available_splits": ["test"],
            "main_score": "cosine_spearman",
            "min_score": 0,
            "max_score": 4,
        }
