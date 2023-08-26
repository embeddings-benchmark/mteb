from ...abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from ...abstasks.BeIRPLTask import BeIRPLTask


class SCIDOCSPL(AbsTaskRetrieval, BeIRPLTask):
    @property
    def description(self):
        return {
            "name": "SCIDOCS-PL",
            "beir_name": "scidocs-pl",
            "description": (
                "SciDocs, a new evaluation benchmark consisting of seven document-level tasks ranging from citation"
                " prediction, to document classification and recommendation."
            ),
            "reference": "https://allenai.org/data/scidocs",
            "benchmark": "BEIR-PL: Zero Shot Information Retrieval Benchmark for the Polish Language",
            "type": "Retrieval",
            "category": "s2p",
            "eval_splits": ["test"],
            "eval_langs": ["en"],
            "main_score": "ndcg_at_10",
        }
