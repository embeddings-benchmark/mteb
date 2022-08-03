from ...abstasks.AbsTaskReranking import AbsTaskReranking


class MindSmallReranking(AbsTaskReranking):
    @property
    def description(self):
        return {
            "name": "MindSmallReranking",
            "hf_hub_name": "mteb/mind_small",
            "description": "Microsoft News Dataset: A Large-Scale English Dataset for News Recommendation Research",
            "reference": "https://www.microsoft.com/en-us/research/uploads/prod/2019/03/nl4se18LinkSO.pdf",
            "type": "Reranking",
            "category": "s2s",
            "eval_splits": ["test"],
            "eval_langs": ["en"],
            "main_score": "map",
        }
