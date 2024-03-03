from ...abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class NFCorpusPL(AbsTaskRetrieval):
    @property
    def description(self):
        return {
            "name": "NFCorpus-PL",
            "hf_hub_name": "clarin-knext/nfcorpus-pl",
            "description": "NFCorpus: A Full-Text Learning to Rank Dataset for Medical Information Retrieval",
            "reference": "https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/",
            "benchmark": "BEIR-PL: Zero Shot Information Retrieval Benchmark for the Polish Language",
            "type": "Retrieval",
            "category": "s2p",
            "eval_splits": ["test"],
            "eval_langs": ["pl"],
            "main_score": "ndcg_at_10",
            "revision": "9a6f9567fda928260afed2de480d79c98bf0bec0",
        }
