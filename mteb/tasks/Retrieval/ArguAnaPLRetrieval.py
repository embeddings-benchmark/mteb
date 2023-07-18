from ...abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from ...abstasks.BeIRPLTask import BeIRPLTask


class ArguAnaPL(AbsTaskRetrieval, BeIRPLTask):
    @property
    def description(self):
        return {
            "name": "ArguAna-PL",
            "beir_name": "arguana-pl",
            "description": "NFCorpus: A Full-Text Learning to Rank Dataset for Medical Information Retrieval",
            "reference": "http://argumentation.bplaced.net/arguana/data",
            "benchmark": "BEIR-PL: Zero Shot Information Retrieval Benchmark for the Polish Language",
            "type": "Retrieval",
            "category": "p2p",
            "eval_splits": ["test"],
            "eval_langs": ["en"],
            "main_score": "ndcg_at_10",
        }
