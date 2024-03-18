from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class CQADupstackAndroidRetrieval(AbsTaskRetrieval):
    @property
    def description(self):
        return {
            "name": "CQADupstackAndroidRetrieval",
            "hf_hub_name": "mteb/cqadupstack-android",
            "description": "CQADupStack: A Benchmark Data Set for Community Question-Answering Research",
            "reference": "http://nlp.cis.unimelb.edu.au/resources/cqadupstack/",
            "type": "Retrieval",
            "category": "s2p",
            "eval_splits": ["test"],
            "eval_langs": ["en"],
            "main_score": "ndcg_at_10",
            "revision": "f46a197baaae43b4f621051089b82a364682dfeb"
        }
