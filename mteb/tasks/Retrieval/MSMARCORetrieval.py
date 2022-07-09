from ...abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from ...abstasks.BeIRTask import BeIRTask


class MSMARCO(AbsTaskRetrieval, BeIRTask):
    @property
    def description(self):
        return {
            "name": "MSMARCO",
            "beir_name": "msmarco",
            "description": "MS MARCO is a collection of datasets focused on deep learning in search",
            "reference": "https://microsoft.github.io/msmarco/",
            "type": "Retrieval",
            "category": "s2s",
            "eval_splits": ["dev", "test"],
            "eval_langs": ["en"],
            "main_score": "ndcg_at_10",
        }
