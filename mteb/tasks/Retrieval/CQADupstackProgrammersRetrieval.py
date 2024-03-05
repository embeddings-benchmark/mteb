from ...abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class CQADupstackProgrammersRetrieval(AbsTaskRetrieval):
    @property
    def description(self):
        return {
            "name": "CQADupstackProgrammersRetrieval",
            "hf_hub_name": "mteb/cqadupstack-programmers",
            "description": "CQADupStack: A Benchmark Data Set for Community Question-Answering Research",
            "reference": "http://nlp.cis.unimelb.edu.au/resources/cqadupstack/",
            "type": "Retrieval",
            "category": "s2p",
            "eval_splits": ["test"],
            "eval_langs": ["en"],
            "main_score": "ndcg_at_10",
            "revision": "6184bc1440d2dbc7612be22b50686b8826d22b32",            
        }
