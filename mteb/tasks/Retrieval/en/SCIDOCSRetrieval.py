from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class SCIDOCS(AbsTaskRetrieval):
    metadata = 

@property
    def metadata_dict(self) -> dict[str, str]:
        return dict(self.metadata)
        return {
            "name": "SCIDOCS",
            "hf_hub_name": "mteb/scidocs",
            "description": (
                "SciDocs, a new evaluation benchmark consisting of seven document-level tasks ranging from citation"
                " prediction, to document classification and recommendation."
            ),
            "reference": "https://allenai.org/data/scidocs",
            "type": "Retrieval",
            "category": "s2p",
            "eval_splits": ["test"],
            "eval_langs": ["en"],
            "main_score": "ndcg_at_10",
        }
