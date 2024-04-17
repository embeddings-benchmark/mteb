import logging
from typing import Dict, List

from .RetrievalEvaluator import (
    RetrievalEvaluator,
)

logger = logging.getLogger(__name__)



class InstructionRetrievalEvaluator(RetrievalEvaluator):
    # only added to extend the RetrievalEvaluator to pass along the instructions
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
