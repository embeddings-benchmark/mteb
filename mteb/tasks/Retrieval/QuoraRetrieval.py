from ...abstasks.AbsTaskRetrieval import AbsTaskRetrieval

class QuoraRetrieval(AbsTaskRetrieval):
    @property
    def description(self):
        return {
            "name": "QuoraRetrieval",
            "hf_hub_name": "mteb/quora-retrieval",
            "description": "QuoraRetrieval is based on questions that are marked as duplicates on the Quora platform. Given a question, find other (duplicate) questions.",
            "reference": "https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs",
            "type": "retrieval",
            "category": None,
            "available_splits": ["test", "validation"],
            "main_score": "map",
        }