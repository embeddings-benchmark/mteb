from ...abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from ...abstasks.BeIRPLTask import BeIRPLTask


class SciFactPL(AbsTaskRetrieval, BeIRPLTask):
    @property
    def description(self):
        return {
            "name": "SciFact-PL",
            "beir_name": "scifact-pl",
            "reference": "https://github.com/allenai/scifact",
            "benchmark": "BEIR-PL: Zero Shot Information Retrieval Benchmark for the Polish Language",
            "description": "SciFact verifies scientific claims using evidence from the research literature containing scientific paper abstracts.",
            "type": "Retrieval",
            "category": "s2p",
            "eval_splits": ["test"],
            "eval_langs": ["pl"],
            "main_score": "ndcg_at_10",
        }
