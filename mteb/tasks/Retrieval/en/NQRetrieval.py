from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class NQ(AbsTaskRetrieval):
    @property
    def metadata_dict(self):
        return {
            "name": "NQ",
            "hf_hub_name": "mteb/nq",
            "description": "NFCorpus: A Full-Text Learning to Rank Dataset for Medical Information Retrieval",
            "reference": "https://ai.google.com/research/NaturalQuestions/",
            "type": "Retrieval",
            "category": "s2p",
            "eval_splits": ["test"],
            "eval_langs": ["en"],
            "main_score": "ndcg_at_10",
            "revision": "b774495ed302d8c44a3a7ea25c90dbce03968f31",
        }
