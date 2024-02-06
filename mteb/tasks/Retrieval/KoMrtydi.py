from ...abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from ...abstasks.BeIRKOTask import BeIRKOTask


class KoMrtydi(AbsTaskRetrieval, BeIRKOTask):
    @property
    def description(self):
        return {
            "name": "Ko-mrtydi",
            "hf_repo": "taeminlee/Ko-mrtydi",
            "hf_repo_qrels": "taeminlee/Ko-mrtydi",
            "beir_name": "Ko-mrtydi",
            "description": "Ko-mrtydi",
            "reference": "",
            "type": "Retrieval",
            "category": "s2p",
            "eval_splits": ["dev"],
            "eval_langs": ["ko"],
            "main_score": "ndcg_at_10",
        }
