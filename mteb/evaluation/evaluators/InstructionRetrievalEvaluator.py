from __future__ import annotations

import logging

from .RetrievalEvaluator import (
    RetrievalEvaluator,
)

logger = logging.getLogger(__name__)


class InstructionRetrievalEvaluator(RetrievalEvaluator):
    # only added to extend the RetrievalEvaluator to pass along the instructions
    def __call__(
        self,
        corpus: dict[str, dict[str, str]],
        queries: dict[str, str],
        instructions: dict[str, str],
        qid: str | None = None,
        **kwargs,
    ) -> dict[str, dict[str, float]]:
        if not self.retriever:
            raise ValueError("Model/Technique has not been provided!")

        if self.is_cross_encoder:
            return self.retriever.search_cross_encoder(
                corpus, queries, self.top_k, instructions=instructions, **kwargs
            )
        elif (
            hasattr(self.retriever.model, "mteb_model_meta")
            and self.retriever.model.mteb_model_meta.name == "bm25s"
        ):
            return self.retriever.model.search(
                corpus,
                queries,
                self.top_k,
                task_name=self.task_name,  # type: ignore
                instructions=instructions,
                **kwargs,
            )
        else:
            return self.retriever.search(
                corpus,
                queries,
                self.top_k,
                instructions=instructions,
                request_qid=qid,
                task_name=self.task_name,
                **kwargs,
            )
