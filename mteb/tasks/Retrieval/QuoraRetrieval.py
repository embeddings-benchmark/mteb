from ...abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from ...abstasks.BeIRTask import BeIRTask

class QuoraRetrieval(AbsTaskRetrieval, BeIRTask):
    @property
    def description(self):
        return {
            "name": "QuoraRetrieval",
            "beir_name": "quora",
            "description": "QuoraRetrieval is based on questions that are marked as duplicates on the Quora platform. Given a question, find other (duplicate) questions.",
            "reference": "https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs",
            "type": "Retrieval",
            "category": "s2s",
            "eval_splits": ["test", "dev"],
            "eval_langs": ["en"],
            "main_score": "map",
        }
