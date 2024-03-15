from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class CQADupstackPhysicsRetrieval(AbsTaskRetrieval):
    @property
    def description(self):
        return {
            "name": "CQADupstackPhysicsRetrieval",
            "hf_hub_name": "mteb/cqadupstack-physics",
            "description": "CQADupStack: A Benchmark Data Set for Community Question-Answering Research",
            "reference": "http://nlp.cis.unimelb.edu.au/resources/cqadupstack/",
            "type": "Retrieval",
            "category": "s2p",
            "eval_splits": ["test"],
            "eval_langs": ["en"],
            "main_score": "ndcg_at_10",
            "revision": "79531abbd1fb92d06c6d6315a0cbbbf5bb247ea4",            
        }
