from ...abstasks import AbsTaskSummarization


class SummEvalSummarization(AbsTaskSummarization):
    @property
    def description(self):
        return {
            "name": "SummEval",
            "hf_hub_name": "mteb/summeval",
            "description": "Biomedical Semantic Similarity Estimation.",
            "reference": "https://tabilab.cmpe.boun.edu.tr/BIOSSES/DataSet.html",
            "type": "Summarization",
            "category": "p2p",
            "eval_splits": ["test"],
            "eval_langs": ["en"],
            "main_score": "cosine_spearman",
            "min_score": 0,
            "max_score": 5,
            "revision": "cda12ad7615edc362dbf25a00fdd61d3b1eaf93c",
        }
