from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class FiQA2018(AbsTaskRetrieval):
    @property
    def description(self):
        return {
            "name": "FiQA2018",
            "hf_hub_name": "mteb/fiqa",
            "description": "Financial Opinion Mining and Question Answering",
            "reference": "https://sites.google.com/view/fiqa/",
            "type": "Retrieval",
            "category": "s2p",
            "eval_splits": ["train", "dev", "test"],
            "eval_langs": ["en"],
            "main_score": "ndcg_at_10",
            "revision": "27a168819829fe9bcd655c2df245fb19452e8e06",            
        }
