from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class ArguAna(AbsTaskRetrieval):
    metadata = TaskMetadata()

    @property
    def metadata_dict(self) -> dict[str, str]:
        return dict(self.metadata)
        return {
            "name": "ArguAna",
            "hf_hub_name": "mteb/arguana",
            "description": "NFCorpus: A Full-Text Learning to Rank Dataset for Medical Information Retrieval",
            "reference": "http://argumentation.bplaced.net/arguana/data",
            "type": "Retrieval",
            "category": "p2p",
            "eval_splits": ["test"],
            "eval_langs": ["en"],
            "main_score": "ndcg_at_10",
            "revision": "c22ab2a51041ffd869aaddef7af8d8215647e41a",
        }
