from ...abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from ...abstasks.BeIRTask import BeIRTask


class MSMARCOv2(AbsTaskRetrieval, BeIRTask):
    @property
    def description(self):
        return {
            "name": "MSMARCOv2",
            "beir_name": "msmarco-v2",
            "description": "MS MARCO is a collection of datasets focused on deep learning in search",
            "reference": "https://microsoft.github.io/msmarco/TREC-Deep-Learning.html",
            "type": "Retrieval",
            "category": "s2p",
            "eval_splits": ["dev", "dev2"],
            "eval_langs": ["en"],
            "main_score": "ndcg_at_10",
        }
