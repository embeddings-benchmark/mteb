from ...abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from ...abstasks.BeIRTask import BeIRTask


class TRECCOVID(AbsTaskRetrieval, BeIRTask):
    @property
    def description(self):
        return {
            "name": "TRECCOVID",
            "beir_name": "trec-covid",
            "description": "TRECCOVID is an ad-hoc search challenge based on the CORD-19 dataset containing scientific articles related to the COVID-19 pandemic",
            "reference": "https://ir.nist.gov/covidSubmit/index.html",
            "description": "TRECCOVID is an ad-hoc search challenge based on the CORD-19 dataset containing scientific articles related to the COVID-19 pandemic.",
            "type": "Retrieval",
            "category": "s2s",
            "eval_splits": ["test"],
            "eval_langs": ["en"],
            "main_score": "ndcg_at_10",
        }
