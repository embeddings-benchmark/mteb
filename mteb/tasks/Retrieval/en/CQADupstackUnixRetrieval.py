from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class CQADupstackUnixRetrieval(AbsTaskRetrieval):
    @property
    def description(self):
        return {
            "name": "CQADupstackUnixRetrieval",
            "hf_hub_name": "mteb/cqadupstack-unix",
            "description": "CQADupStack: A Benchmark Data Set for Community Question-Answering Research",
            "reference": "http://nlp.cis.unimelb.edu.au/resources/cqadupstack/",
            "type": "Retrieval",
            "category": "s2p",
            "eval_splits": ["test"],
            "eval_langs": ["en"],
            "main_score": "ndcg_at_10",
            "revision": "6c6430d3a6d36f8d2a829195bc5dc94d7e063e53",            
        }
