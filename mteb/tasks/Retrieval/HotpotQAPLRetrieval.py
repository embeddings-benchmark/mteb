from ...abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from ...abstasks.BeIRPLTask import BeIRPLTask


class HotpotQAPL(AbsTaskRetrieval, BeIRPLTask):
    @property
    def description(self):
        return {
            "name": "HotpotQA-PL",
            "beir_name": "hotpotqa-pl",
            "description": (
                "HotpotQA is a question answering dataset featuring natural, multi-hop questions, with strong"
                " supervision for supporting facts to enable more explainable question answering systems."
            ),
            "reference": "https://hotpotqa.github.io/",
            "benchmark": "BEIR-PL: Zero Shot Information Retrieval Benchmark for the Polish Language",
            "type": "Retrieval",
            "category": "s2p",
            "eval_splits": ["test"],
            "eval_langs": ["pl"],
            "main_score": "ndcg_at_10",
        }
