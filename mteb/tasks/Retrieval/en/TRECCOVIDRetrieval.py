from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class TRECCOVID(AbsTaskRetrieval):
    metadata = 

@property
    def metadata_dict(self) -> dict[str, str]:
        return dict(self.metadata)
        return {
            "name": "TRECCOVID",
            "hf_hub_name": "mteb/trec-covid",
            "description": "TRECCOVID is an ad-hoc search challenge based on the CORD-19 dataset containing scientific articles related to the COVID-19 pandemic",
            "reference": "https://ir.nist.gov/covidSubmit/index.html",
            "description": "TRECCOVID is an ad-hoc search challenge based on the CORD-19 dataset containing scientific articles related to the COVID-19 pandemic.",
            "type": "Retrieval",
            "category": "s2p",
            "eval_splits": ["test"],
            "eval_langs": ["en"],
            "main_score": "ndcg_at_10",
        }
