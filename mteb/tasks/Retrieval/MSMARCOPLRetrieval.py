from ...abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class MSMARCOPL(AbsTaskRetrieval):
    @property
    def description(self):
        return {
            "name": "MSMARCO-PL",
            "hf_hub_name": "clarin-knext/msmarco-pl",
            "description": "MS MARCO is a collection of datasets focused on deep learning in search",
            "reference": "https://microsoft.github.io/msmarco/",
            "benchmark": "BEIR-PL: Zero Shot Information Retrieval Benchmark for the Polish Language",
            "type": "Retrieval",
            "category": "s2p",
            "eval_splits": ["validation", "test"],  # "validation" if using latest BEIR i.e. HFDataLoader
            "eval_langs": ["pl"],
            "main_score": "ndcg_at_10",
            "revision": "8634c07806d5cce3a6138e260e59b81760a0a640",
        }
