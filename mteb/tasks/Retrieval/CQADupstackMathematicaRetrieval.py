from ...abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from ...abstasks.BeIRTask import BeIRTask


class CQADupstackMathematicaRetrieval(AbsTaskRetrieval, BeIRTask):
    @property
    def description(self):
        return {
            "name": "CQADupstackMathematicaRetrieval",
            "beir_name": "cqadupstack/mathematica",
            "description": "CQADupStack: A Benchmark Data Set for Community Question-Answering Research",
            "reference": "http://nlp.cis.unimelb.edu.au/resources/cqadupstack/",
            "type": "Retrieval",
            "category": "s2p",
            "eval_splits": ["test"],
            "eval_langs": ["en"],
            "main_score": "ndcg_at_10",
        }
