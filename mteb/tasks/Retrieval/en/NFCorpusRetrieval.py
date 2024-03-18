from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class NFCorpus(AbsTaskRetrieval):
    @property
    def description(self):
        return {
            "name": "NFCorpus",
            "hf_hub_name": "mteb/nfcorpus",
            "description": "NFCorpus: A Full-Text Learning to Rank Dataset for Medical Information Retrieval",
            "reference": "https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/",
            "type": "Retrieval",
            "category": "s2p",
            "eval_splits": ["test"],
            "eval_langs": ["en"],
            "main_score": "ndcg_at_10",
            "revision": "ec0fa4fe99da2ff19ca1214b7966684033a58814",
        }
