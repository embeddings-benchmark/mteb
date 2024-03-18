from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class MSMARCO(AbsTaskRetrieval):
    @property
    def description(self):
        return {
            "name": "MSMARCO",
            "hf_hub_name": "mteb/msmarco",
            "description": "MS MARCO is a collection of datasets focused on deep learning in search",
            "reference": "https://microsoft.github.io/msmarco/",
            "type": "Retrieval",
            "category": "s2p",
            "eval_splits": ["train", "dev", "test"],
            "eval_langs": ["en"],
            "main_score": "ndcg_at_10",
            "revision": "c5a29a104738b98a9e76336939199e264163d4a0",
        }
