from ...abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class NQPL(AbsTaskRetrieval):
    @property
    def description(self):
        return {
            "name": "NQ-PL",
            "hf_hub_name": "clarin-knext/nq-pl",
            "description": "Natural Questions: A Benchmark for Question Answering Research",
            "reference": "https://ai.google.com/research/NaturalQuestions/",
            "benchmark": "BEIR-PL: Zero Shot Information Retrieval Benchmark for the Polish Language",
            "type": "Retrieval",
            "category": "s2p",
            "eval_splits": ["test"],
            "eval_langs": ["pl"],
            "main_score": "ndcg_at_10",
            "revision": "f171245712cf85dd4700b06bef18001578d0ca8d",
        }
