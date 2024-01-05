from ...abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from ...abstasks.BeIRDETask import BeIRDETask


class GermanRetrieval(AbsTaskRetrieval, BeIRDETask):

    @property
    def description(self):
        return {
            "name": "GermanQuAD-Retrieval",
            "beir_name": "germanquad",
            "description": "Context Retrieval for German Question Answering",
            "reference": "GermanQuAD and GermanDPR: Improving Non-English Question Answering and Passage Retrieval",
            # "benchmark": "", #TODO: figure out what goes into a proper description
            "type": "Retrieval",
            "category": "p2p",
            "eval_splits": ["test"],
            "eval_langs": ["de"],
            "main_score": "mrr_at_10",
        }
