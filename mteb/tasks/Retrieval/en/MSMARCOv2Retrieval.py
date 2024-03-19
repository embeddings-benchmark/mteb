from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class MSMARCOv2(AbsTaskRetrieval):
    @property
    def metadata_dict(self):
        return {
            "name": "MSMARCOv2",
            "hf_hub_name": "mteb/msmarco-v2",
            "description": "MS MARCO is a collection of datasets focused on deep learning in search",
            "reference": "https://microsoft.github.io/msmarco/TREC-Deep-Learning.html",
            "type": "Retrieval",
            "category": "s2p",
            "eval_splits": ["train", "dev", "dev2"],
            "eval_langs": ["en"],
            "main_score": "ndcg_at_10",
            "revision": "b1663124850d305ab7c470bb0548acf8e2e7ea43",
        }
