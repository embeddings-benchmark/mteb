from ...abstasks.AbsTaskSTS import AbsTaskSTS


class SickrSTS(AbsTaskSTS):
    @property
    def description(self):
        return {
            "name": "SICK-R",
            "hf_hub_name": "mteb/biosses-sts",
            "description": "Semantic Textual Similarity SICK-R dataset as described here:",
            "reference": "https://www.aclweb.org/anthology/S14-2001.pdf",
            "type": "STS",
            "category": "s2s",
            "available_splits": ["train", "validation", "test"],
            "main_score": "cosine_spearman",
            "min_score": 1,
            "max_score": 5,
        }
