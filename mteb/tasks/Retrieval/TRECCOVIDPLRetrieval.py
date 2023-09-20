from ...abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from ...abstasks.BeIRPLTask import BeIRPLTask


class TRECCOVIDPL(AbsTaskRetrieval, BeIRPLTask):
    @property
    def description(self):
        return {
            "name": "TRECCOVID-PL",
            "beir_name": "trec-covid-pl",
            "reference": "https://ir.nist.gov/covidSubmit/index.html",
            "benchmark": "BEIR-PL: Zero Shot Information Retrieval Benchmark for the Polish Language",
            "description": "TRECCOVID is an ad-hoc search challenge based on the CORD-19 dataset containing scientific articles related to the COVID-19 pandemic.",
            "type": "Retrieval",
            "category": "s2p",
            "eval_splits": ["test"],
            "eval_langs": ["pl"],
            "main_score": "ndcg_at_10",
        }
