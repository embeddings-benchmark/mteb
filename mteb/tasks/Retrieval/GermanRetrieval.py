from ...abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from ...abstasks.GermanRetrievalTask import GermanRetrievalTask


class GermanRetrieval(AbsTaskRetrieval, GermanRetrievalTask):

    @property
    def description(self):
        return {
            "name": "GermanQuAD-Retrieval",
            "beir_name": "germanquad-retrieval",
            "description": "Context Retrieval for German Question Answering",
            "reference": "https://www.deepset.ai/germanquad",
            "type": "Retrieval",
            "category": "p2p",
            "eval_splits": ["test"],
            "eval_langs": ["de"],
            "main_score": "mrr_at_10",
        }
