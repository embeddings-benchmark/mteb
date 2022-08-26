from ...abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from ...abstasks.BeIRTask import BeIRTask


class Touche2020(AbsTaskRetrieval, BeIRTask):
    @property
    def description(self):
        return {
            "name": "Touche2020",
            "beir_name": "webis-touche2020",
            "description": "Touché Task 1: Argument Retrieval for Controversial Questions",
            "reference": "https://webis.de/events/touche-20/shared-task-1.html",
            "type": "Retrieval",
            "category": "s2s",
            "eval_splits": ["test"],
            "eval_langs": ["en"],
            "main_score": "ndcg_at_10",
        }
