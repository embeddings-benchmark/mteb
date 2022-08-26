from ...abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from ...abstasks.BeIRTask import BeIRTask


class CQADupstackPhysicsRetrieval(AbsTaskRetrieval, BeIRTask):
    @property
    def description(self):
        return {
            "name": "CQADupstackPhysicsRetrieval",
            "beir_name": "cqadupstack/physics",
            "description": "CQADupStack: A Benchmark Data Set for Community Question-Answering Research",
            "reference": "http://nlp.cis.unimelb.edu.au/resources/cqadupstack/",
            "type": "Retrieval",
            "category": "s2s",
            "eval_splits": ["test"],
            "eval_langs": ["en"],
            "main_score": "ndcg_at_10",
        }
