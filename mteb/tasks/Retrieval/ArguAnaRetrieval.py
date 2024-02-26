from ...abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class ArguAna(AbsTaskRetrieval):
    @property
    def description(self):
        return {
            "name": "ArguAna",
            "beir_name": "arguana",
            "description": "NFCorpus: A Full-Text Learning to Rank Dataset for Medical Information Retrieval",
            "reference": "http://argumentation.bplaced.net/arguana/data",
            "type": "Retrieval",
            "category": "p2p",
            "eval_splits": ["test"],
            "eval_langs": ["en"],
            "main_score": "ndcg_at_10",
        }
