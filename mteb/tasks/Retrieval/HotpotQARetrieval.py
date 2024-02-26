from ...abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class HotpotQA(AbsTaskRetrieval):
    @property
    def description(self):
        return {
            "name": "HotpotQA",
            "beir_name": "hotpotqa",
            "description": (
                "HotpotQA is a question answering dataset featuring natural, multi-hop questions, with strong"
                " supervision for supporting facts to enable more explainable question answering systems."
            ),
            "reference": "https://hotpotqa.github.io/",
            "type": "Retrieval",
            "category": "s2p",
            "eval_splits": ["test"],
            "eval_langs": ["en"],
            "main_score": "ndcg_at_10",
        }
