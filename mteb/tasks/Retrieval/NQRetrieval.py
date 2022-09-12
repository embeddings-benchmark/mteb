from ...abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from ...abstasks.BeIRTask import BeIRTask


class NQ(AbsTaskRetrieval, BeIRTask):
    @property
    def description(self):
        return {
            "name": "NQ",
            "beir_name": "nq",
            "description": "NFCorpus: A Full-Text Learning to Rank Dataset for Medical Information Retrieval",
            "reference": "https://ai.google.com/research/NaturalQuestions/",
            "type": "Retrieval",
            "category": "s2p",
            "eval_splits": ["test"],
            "eval_langs": ["en"],
            "main_score": "ndcg_at_10",
        }
