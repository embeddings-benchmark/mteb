from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class DBPediaPL(AbsTaskRetrieval):
    @property
    def description(self):
        return {
            "name": "DBPedia-PL",
            "hf_hub_name": "clarin-knext/dbpedia-pl",
            "description": (
                "DBpedia-Entity is a standard test collection for entity search over the DBpedia knowledge base"
            ),
            "reference": "https://github.com/iai-group/DBpedia-Entity/",
            "benchmark": "BEIR-PL: Zero Shot Information Retrieval Benchmark for the Polish Language",
            "type": "Retrieval",
            "category": "s2p",
            "eval_splits": ["test"],
            "eval_langs": ["pl"],
            "main_score": "ndcg_at_10",
            "revision": "76afe41d9af165cc40999fcaa92312b8b012064a",            
        }
