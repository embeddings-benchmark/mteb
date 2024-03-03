from ...abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class CQADupstackWebmastersRetrieval(AbsTaskRetrieval):
    @property
    def description(self):
        return {
            "name": "CQADupstackWebmastersRetrieval",
            "hf_hub_name": "mteb/cqadupstack-webmasters",
            "description": "CQADupStack: A Benchmark Data Set for Community Question-Answering Research",
            "reference": "http://nlp.cis.unimelb.edu.au/resources/cqadupstack/",
            "type": "Retrieval",
            "category": "s2p",
            "eval_splits": ["test"],
            "eval_langs": ["en"],
            "main_score": "ndcg_at_10",
            "revision": "160c094312a0e1facb97e55eeddb698c0abe3571",            
        }
