import logging
from typing import Dict, List

from .RetrievalEvaluator import (
    DenseRetrievalExactSearch,
    DRESModel,
    RetrievalEvaluator,
    is_dres_compatible,
)

logger = logging.getLogger(__name__)


# Adapted from https://github.com/beir-cellar/beir/blob/f062f038c4bfd19a8ca942a9910b1e0d218759d4/beir/retrieval/evaluation.py#L9
class InstructionRetrievalEvaluator(RetrievalEvaluator):
    def __init__(
        self,
        retriever,
        k_values: List[int] = [1, 3, 5, 10, 20, 100, 1000],
        score_function: str = "cos_sim",
        **kwargs,
    ):
        super().__init__(**kwargs)
        if is_dres_compatible(retriever):
            logger.info(
                "The custom encode_queries and encode_corpus functions of the model will be used"
            )
            self.retriever = DenseRetrievalExactSearch(retriever, **kwargs)
        else:
            self.retriever = DenseRetrievalExactSearch(DRESModel(retriever), **kwargs)
        self.k_values = k_values
        self.top_k = max(k_values)
        self.score_function = score_function

    def __call__(
        self,
        corpus: Dict[str, Dict[str, str]],
        queries: Dict[str, str],
        instructions: Dict[str, str],
        **kwargs,
    ) -> Dict[str, Dict[str, float]]:
        if not self.retriever:
            raise ValueError("Model/Technique has not been provided!")
        return self.retriever.search(
            corpus,
            queries,
            self.top_k,
            self.score_function,
            instructions=instructions,
            **kwargs,
        )
