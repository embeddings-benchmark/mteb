from ...abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from ...abstasks.BeIRPLTask import BeIRPLTask


class QuoraPLRetrieval(AbsTaskRetrieval, BeIRPLTask):
    @property
    def description(self):
        return {
            "name": "Quora-PL",
            "beir_name": "quora-pl",
            "description": (
                "QuoraRetrieval is based on questions that are marked as duplicates on the Quora platform. Given a"
                " question, find other (duplicate) questions."
            ),
            "reference": "https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs",
            "type": "Retrieval",
            "benchmark": "BEIR-PL: Zero Shot Information Retrieval Benchmark for the Polish Language",
            "category": "s2s",
            "eval_splits": ["validation", "test"],  # validation for new DataLoader
            "eval_langs": ["en"],
            "main_score": "ndcg_at_10",
        }
