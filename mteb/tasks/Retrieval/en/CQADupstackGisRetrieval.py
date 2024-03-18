from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class CQADupstackGisRetrieval(AbsTaskRetrieval):
    @property
    def description(self):
        return {
            "name": "CQADupstackGisRetrieval",
            "hf_hub_name": "mteb/cqadupstack-gis",
            "description": "CQADupStack: A Benchmark Data Set for Community Question-Answering Research",
            "reference": "http://nlp.cis.unimelb.edu.au/resources/cqadupstack/",
            "type": "Retrieval",
            "category": "s2p",
            "eval_splits": ["test"],
            "eval_langs": ["en"],
            "main_score": "ndcg_at_10",
            "revision": "5003b3064772da1887988e05400cf3806fe491f2",            
        }
