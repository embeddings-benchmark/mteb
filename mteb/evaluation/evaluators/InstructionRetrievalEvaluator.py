import logging
from typing import Dict, Union

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
        qid: Union[str, None] = None,
        **kwargs,
    ) -> Dict[str, Dict[str, float]]:
        if not self.retriever:
            raise ValueError("Model/Technique has not been provided!")

        if self.is_cross_encoder:
            return self.retriever.search_cross_encoder(
                corpus, queries, self.top_k, instructions=instructions, **kwargs
            )
        else:
            return self.retriever.search(
                corpus,
                queries,
                self.top_k,
                self.score_function,
                instructions=instructions,
                request_qid=qid,
                prompt_name=self.task_name,
                **kwargs,
            )
