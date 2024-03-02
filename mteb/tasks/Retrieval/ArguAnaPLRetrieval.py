from ...abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class ArguAnaPL(AbsTaskRetrieval):
    @property
    def description(self):
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
            "revision": "",
        }
