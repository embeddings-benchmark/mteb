from ...abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from ...abstasks.BeIRTask import BeIRTask


class CQADupstackUnixRetrieval(AbsTaskRetrieval, BeIRTask):
    @property
    def description(self):
        return {
            "name": "CQADupstackUnixRetrieval",
            "beir_name": "cqadupstack/unix",
            "description": "CQADupStack: A Benchmark Data Set for Community Question-Answering Research",
            "reference": "http://nlp.cis.unimelb.edu.au/resources/cqadupstack/",
            "type": "Retrieval",
            "category": "s2p",
            "eval_splits": ["test"],
            "eval_langs": ["en"],
            "main_score": "ndcg_at_10",
        }
