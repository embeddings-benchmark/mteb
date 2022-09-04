from ...abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from ...abstasks.BeIRTask import BeIRTask


class SciFact(AbsTaskRetrieval, BeIRTask):
    @property
    def description(self):
        return {
            "name": "SciFact",
            "beir_name": "scifact",
            "reference": "https://github.com/allenai/scifact",
            "description": "SciFact verifies scientific claims using evidence from the research literature containing scientific paper abstracts.",
            "type": "Retrieval",
            "category": "s2s",
            "eval_splits": ["test"],
            "eval_langs": ["en"],
            "main_score": "ndcg_at_10",
        }
