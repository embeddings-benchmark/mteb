from ...abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from ...abstasks.BeIRTask import BeIRTask


class SCIDOCS(AbsTaskRetrieval, BeIRTask):
    @property
    def description(self):
        return {
            "name": "SCIDOCS",
            "beir_name": "scidocs",
            "description": (
                "SciDocs, a new evaluation benchmark consisting of seven document-level tasks ranging from citation"
                " prediction, to document classification and recommendation."
            ),
            "reference": "https://allenai.org/data/scidocs",
            "type": "Retrieval",
            "category": "s2p",
            "eval_splits": ["test"],
            "eval_langs": ["en"],
            "main_score": "ndcg_at_10",
        }
