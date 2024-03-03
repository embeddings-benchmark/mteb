from ...abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class Touche2020(AbsTaskRetrieval):
    @property
    def description(self):
        return {
            "name": "Touche2020",
            "hf_hub_name": "mteb/touche2020",
            "description": "Touché Task 1: Argument Retrieval for Controversial Questions",
            "reference": "https://webis.de/events/touche-20/shared-task-1.html",
            "type": "Retrieval",
            "category": "s2p",
            "eval_splits": ["test"],
            "eval_langs": ["en"],
            "main_score": "ndcg_at_10",
            "revision": "",
        }
