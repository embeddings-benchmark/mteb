from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class CQADupstackStatsRetrieval(AbsTaskRetrieval):
    @property
    def description(self):
        return {
            "name": "CQADupstackStatsRetrieval",
            "hf_hub_name": "mteb/cqadupstack-stats",
            "description": "CQADupStack: A Benchmark Data Set for Community Question-Answering Research",
            "reference": "http://nlp.cis.unimelb.edu.au/resources/cqadupstack/",
            "type": "Retrieval",
            "category": "s2p",
            "eval_splits": ["test"],
            "eval_langs": ["en"],
            "main_score": "ndcg_at_10",
            "revision": "65ac3a16b8e91f9cee4c9828cc7c335575432a2a",            
        }
