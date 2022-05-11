from ...abstasks.AbsTaskReranking import AbsTaskReranking


class SciDocsReranking(AbsTaskReranking):
    @property
    def description(self):
        return {
            "name": "SciDocs",
            "hf_hub_name": "mteb/scidocs-reranking",
            "description": "Ranking of related scientific papers based on their title.",
            "reference": "https://allenai.org/data/scidocs",
            "type": "Reranking",
            "category": "s2s",
            "available_splits": ["test", "validation"],
            "main_score": "map",
        }
