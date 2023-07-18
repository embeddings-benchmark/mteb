from ...abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from ...abstasks.BeIRPLTask import BeIRPLTask


class DBPediaPL(AbsTaskRetrieval, BeIRPLTask):
    @property
    def description(self):
        return {
            "name": "DBPedia-pl",
            "beir_name": "dbpedia-pl",
            "description": (
                "DBpedia-Entity is a standard test collection for entity search over the DBpedia knowledge base"
            ),
            "reference": "https://github.com/iai-group/DBpedia-Entity/",
            "benchmark": "BEIR-PL: Zero Shot Information Retrieval Benchmark for the Polish Language",
            "type": "Retrieval",
            "category": "s2p",
            "eval_splits": ["test"],
            "eval_langs": ["en"],
            "main_score": "ndcg_at_10",
        }
