from ...abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from ...abstasks.BeIRKOTask import BeIRKOTask


class KoMiracl(AbsTaskRetrieval, BeIRKOTask):
    @property
    def description(self):
        return {
            "name": "Ko-miracl",
            "hf_repo": "taeminlee/Ko-miracl",
            "hf_repo_qrels": "taeminlee/Ko-miracl",
            "beir_name": "Ko-miracl",
            "description": "Ko-miracl",
            "reference": "",
            "type": "Retrieval",
            "category": "s2p",
            "eval_splits": ["dev"],
            "eval_langs": ["ko"],
            "main_score": "ndcg_at_10",
        }
