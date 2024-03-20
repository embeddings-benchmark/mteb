from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class ArguAnaPL(AbsTaskRetrieval):
    metadata = TaskMetadata()

    @property
    def metadata_dict(self) -> dict[str, str]:
        return dict(self.metadata)
        return {
            "name": "ArguAna-PL",
            "hf_hub_name": "clarin-knext/arguana-pl",
            "description": "NFCorpus: A Full-Text Learning to Rank Dataset for Medical Information Retrieval",
            "reference": "http://argumentation.bplaced.net/arguana/data",
            "benchmark": "BEIR-PL: Zero Shot Information Retrieval Benchmark for the Polish Language",
            "type": "Retrieval",
            "category": "p2p",
            "eval_splits": ["test"],
            "eval_langs": ["pl"],
            "main_score": "ndcg_at_10",
            "revision": "63fc86750af76253e8c760fc9e534bbf24d260a2",
        }
