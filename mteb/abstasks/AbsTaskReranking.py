from ..evaluation.evaluators import RerankingEvaluator
from .AbsTask import AbsTask


class AbsTaskReranking(AbsTask):
    """
    Abstract class for re-ranking experiments.

    self.load_data() must generate a huggingface dataset with a split matching self.metadata_dict["eval_splits"], and assign it to self.dataset. It must contain the following columns:
        query: str
        positive: list[str]
        negative: list[str]
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def evaluate(self, model, split="test", **kwargs):
        if not self.data_loaded:
            self.load_data()

        scores = {}
        if self.is_multilingual:
            for lang in self.langs:
                data_split = self.dataset[lang][split]
                evaluator = RerankingEvaluator(data_split, **kwargs)
                scores[lang] = evaluator(model)
        else:
            data_split = self.dataset[split]

            evaluator = RerankingEvaluator(data_split, **kwargs)
            scores = evaluator(model)

        return dict(scores)
