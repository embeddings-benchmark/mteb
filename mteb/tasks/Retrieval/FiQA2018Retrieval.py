from ...abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class FiQA2018(AbsTaskRetrieval):
    @property
    def description(self):
        return {
            "name": "FiQA2018",
            "beir_name": "fiqa",
            "description": "Financial Opinion Mining and Question Answering",
            "reference": "https://sites.google.com/view/fiqa/",
            "type": "Retrieval",
            "category": "s2p",
            "eval_splits": ["test"],
            "eval_langs": ["en"],
            "main_score": "ndcg_at_10",
        }
