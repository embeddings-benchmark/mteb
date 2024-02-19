from ...abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from ...abstasks.BeIRKOTask import BeIRKOTask


class KoStrategyQA(AbsTaskRetrieval, BeIRKOTask):
    @property
    def description(self):
        return {
            "name": "Ko-StrategyQA",
            "hf_repo": "taeminlee/Ko-StrategyQA",
            "hf_repo_qrels": "taeminlee/Ko-StrategyQA",
            "beir_name": "Ko-StrategyQA",
            "description": "Ko-StrategyQA",
            "reference": "",
            "type": "Retrieval",
            "category": "s2p",
            "eval_splits": ["dev"],
            "eval_langs": ["ko"],
            "main_score": "ndcg_at_10",
        }
