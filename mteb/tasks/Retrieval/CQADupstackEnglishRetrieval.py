from ...abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class CQADupstackEnglishRetrieval(AbsTaskRetrieval):
    @property
    def description(self):
        return {
            "name": "CQADupstackEnglishRetrieval",
            "hf_hub_name": "mteb/cqadupstack-english",
            "description": "CQADupStack: A Benchmark Data Set for Community Question-Answering Research",
            "reference": "http://nlp.cis.unimelb.edu.au/resources/cqadupstack/",
            "type": "Retrieval",
            "category": "s2p",
            "eval_splits": ["test"],
            "eval_langs": ["en"],
            "main_score": "ndcg_at_10",
            "revision": "",            
        }
