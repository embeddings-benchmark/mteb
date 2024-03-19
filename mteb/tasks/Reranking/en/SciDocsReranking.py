from ....abstasks.AbsTaskReranking import AbsTaskReranking


class SciDocsReranking(AbsTaskReranking):
    @property
    def metadata_dict(self):
        return {
            "name": "SciDocsRR",
            "hf_hub_name": "mteb/scidocs-reranking",
            "description": "Ranking of related scientific papers based on their title.",
            "reference": "https://allenai.org/data/scidocs",
            "type": "Reranking",
            "category": "s2s",
            "eval_splits": ["test", "validation"],
            "eval_langs": ["en"],
            "main_score": "map",
            "revision": "d3c5e1fc0b855ab6097bf1cda04dd73947d7caab",
        }
