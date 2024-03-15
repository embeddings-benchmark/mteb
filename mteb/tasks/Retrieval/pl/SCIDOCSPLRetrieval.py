from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class SCIDOCSPL(AbsTaskRetrieval):
    @property
    def description(self):
        return {
            "name": "SCIDOCS-PL",
            "hf_hub_name": "clarin-knext/scidocs-pl",
            "description": (
                "SciDocs, a new evaluation benchmark consisting of seven document-level tasks ranging from citation"
                " prediction, to document classification and recommendation."
            ),
            "reference": "https://allenai.org/data/scidocs",
            "benchmark": "BEIR-PL: Zero Shot Information Retrieval Benchmark for the Polish Language",
            "type": "Retrieval",
            "category": "s2p",
            "eval_splits": ["test"],
            "eval_langs": ["pl"],
            "main_score": "ndcg_at_10",
            "revision": "45452b03f05560207ef19149545f168e596c9337",
        }
