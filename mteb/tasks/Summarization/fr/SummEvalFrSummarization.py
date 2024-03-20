from mteb.abstasks import AbsTaskSummarization


class SummEvalFrSummarization(AbsTaskSummarization):
    metadata = 

@property
    def metadata_dict(self) -> dict[str, str]:
        return dict(self.metadata)
        return {
            "name": "SummEvalFr",
            "hf_hub_name": "lyon-nlp/summarization-summeval-fr-p2p",
            "description": "News Article Summary Semantic Similarity Estimation translated from english to french with DeepL.",
            "reference": "https://github.com/Yale-LILY/SummEval",
            "type": "Summarization",
            "category": "p2p",
            "eval_splits": ["test"],
            "eval_langs": ["fr"],
            "main_score": "cosine_spearman",
            "min_score": 0,
            "max_score": 5,
            "revision": "b385812de6a9577b6f4d0f88c6a6e35395a94054",
        }
