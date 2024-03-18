from ..evaluation.evaluators import RerankingEvaluator
from .AbsTask import AbsTask


class AbsTaskReranking(AbsTask):
    """
    Abstract class for re-ranking experiments.
    Child-classes must implement the following properties:
    self.corpus = {'dev': Dict[id, str], 'test': Dict[id, str]}         #id => sentence
    self.queries = {'dev': Dict[id, str], 'test': Dict[id, str]}
    self.relevant_docs = {'dev': Dict[id, set], 'test': Dict[id, set]}

    self.load_data() must return a huggingface dataset containing a split matching the task description's "eval_splits" and the following columns:
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
