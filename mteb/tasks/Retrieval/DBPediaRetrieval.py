from ...abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from ...abstasks.BeIRTask import BeIRTask


class DBPedia(AbsTaskRetrieval, BeIRTask):
    @property
    def description(self):
        return {
            "name": "DBPedia",
            "beir_name": "dbpedia-entity",
            "description": (
                "DBpedia-Entity is a standard test collection for entity search over the DBpedia knowledge base"
            ),
            "reference": "https://github.com/iai-group/DBpedia-Entity/",
            "type": "Retrieval",
            "category": "s2p",
            "eval_splits": ["test"],
            "eval_langs": ["en"],
            "main_score": "ndcg_at_10",
            "revision": "f097057d03ed98220bc7309ddb10b71a54d667d6",
        }
