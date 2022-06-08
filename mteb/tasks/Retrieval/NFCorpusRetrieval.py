from ...abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from ...abstasks.BeIRTask import BeIRTask


class NFCorpus(AbsTaskRetrieval, BeIRTask):
    @property
    def description(self):
        return {
            "name": "NFCorpus",
            "beir_name": "nfcorpus",
            "description": "NFCorpus: A Full-Text Learning to Rank Dataset for Medical Information Retrieval",
            "reference": "https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/",
            "type": "Retrieval",
            "category": "s2s",
            "eval_splits": ["train", "dev", "test"],
            "eval_langs": ["en"],
            "main_score": "map",
        }
