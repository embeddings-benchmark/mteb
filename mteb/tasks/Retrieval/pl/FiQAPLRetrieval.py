from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class FiQAPLRetrieval(AbsTaskRetrieval):
    @property
    def description(self):
        return {
            "name": "FiQA-PL",
            "hf_hub_name": "clarin-knext/fiqa-pl",
            "description": "Financial Opinion Mining and Question Answering",
            "reference": "https://sites.google.com/view/fiqa/",
            "benchmark": "BEIR-PL: Zero Shot Information Retrieval Benchmark for the Polish Language",
            "type": "Retrieval",
            "category": "s2p",
            "eval_splits": ["test"],
            "eval_langs": ["pl"],
            "main_score": "ndcg_at_10",
            "revision": "2e535829717f8bf9dc829b7f911cc5bbd4e6608e",            
        }
