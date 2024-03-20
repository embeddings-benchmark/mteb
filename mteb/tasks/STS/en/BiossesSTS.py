from ....abstasks.AbsTaskSTS import AbsTaskSTS


class BiossesSTS(AbsTaskSTS):
    metadata = TaskMetadata()

    @property
    def metadata_dict(self) -> dict[str, str]:
        return dict(self.metadata)
        return {
            "name": "BIOSSES",
            "hf_hub_name": "mteb/biosses-sts",
            "description": "Biomedical Semantic Similarity Estimation.",
            "reference": "https://tabilab.cmpe.boun.edu.tr/BIOSSES/DataSet.html",
            "type": "STS",
            "category": "s2s",
            "eval_splits": ["test"],
            "eval_langs": ["en"],
            "main_score": "cosine_spearman",
            "min_score": 0,
            "max_score": 4,
            "revision": "d3fb88f8f02e40887cd149695127462bbcf29b4a",
        }
