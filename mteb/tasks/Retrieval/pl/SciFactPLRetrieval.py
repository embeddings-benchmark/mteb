from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class SciFactPL(AbsTaskRetrieval):
    @property
    def description(self):
        return {
            "name": "SciFact-PL",
            "hf_hub_name": "clarin-knext/scifact-pl",
            "reference": "https://github.com/allenai/scifact",
            "benchmark": "BEIR-PL: Zero Shot Information Retrieval Benchmark for the Polish Language",
            "description": "SciFact verifies scientific claims using evidence from the research literature containing scientific paper abstracts.",
            "type": "Retrieval",
            "category": "s2p",
            "eval_splits": ["test"],
            "eval_langs": ["pl"],
            "main_score": "ndcg_at_10",
            "revision": "47932a35f045ef8ed01ba82bf9ff67f6e109207e",
        }
