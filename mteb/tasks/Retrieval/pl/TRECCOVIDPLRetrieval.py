from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class TRECCOVIDPL(AbsTaskRetrieval):
    @property
    def description(self):
        return {
            "name": "TRECCOVID-PL",
            "hf_hub_name": "clarin-knext/trec-covid-pl",
            "reference": "https://ir.nist.gov/covidSubmit/index.html",
            "benchmark": "BEIR-PL: Zero Shot Information Retrieval Benchmark for the Polish Language",
            "description": "TRECCOVID is an ad-hoc search challenge based on the CORD-19 dataset containing scientific articles related to the COVID-19 pandemic.",
            "type": "Retrieval",
            "category": "s2p",
            "eval_splits": ["test"],
            "eval_langs": ["pl"],
            "main_score": "ndcg_at_10",
            "revision": "81bcb408f33366c2a20ac54adafad1ae7e877fdd",
        }
