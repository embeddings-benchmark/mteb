from ...abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from ...abstasks.BeIRTask import BeIRTask


class TRECCOVID(AbsTaskRetrieval, BeIRTask):
    @property
    def description(self):
        return {
            "name": "TRECCOVID",
            "beir_name": "trec-covid",
            "reference": "https://ir.nist.gov/covidSubmit/index.html",
            "type": "Retrieval",
            "category": "s2s",
            "eval_splits": ["test"],
            "eval_langs": ["en"],
            "main_score": "map",
        }
