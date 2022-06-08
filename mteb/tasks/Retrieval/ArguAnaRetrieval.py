from ...abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from ...abstasks.BeIRTask import BeIRTask


class ArguAna(AbsTaskRetrieval, BeIRTask):
    @property
    def description(self):
        return {
            "name": "ArguAna",
            "beir_name": "arguana",
            "description": "NFCorpus: A Full-Text Learning to Rank Dataset for Medical Information Retrieval",
            "reference": "http://argumentation.bplaced.net/arguana/data",
            "type": "Retrieval",
            "category": "s2s",
            "eval_splits": ["test"],
            "eval_langs": ["en"],
            "main_score": "map",
        }
