"""Retrieval tasks whose qrels carry continuous relevance gains next to the integer labels.

The qrels config of such a task has two extra columns, ``gain`` (float) and ``theta``. The
standard loader reads only ``query-id``, ``corpus-id`` and ``score``, so every existing metric is
unchanged. This task class additionally reads the ``gain`` column and reports ``ndcg_float_at_k``
(NDCG over the float gains, see :func:`mteb._evaluators.retrieval_metrics.ndcg_float_scores`)
through :meth:`AbsTaskRetrieval.task_specific_scores`.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING, Any

from datasets import load_dataset

from mteb._evaluators.retrieval_metrics import ndcg_float_scores
from mteb.abstasks.retrieval import AbsTaskRetrieval

if TYPE_CHECKING:
    from mteb.types import RelevantDocumentsType

logger = logging.getLogger(__name__)


class AbsTaskRetrievalFloatGains(AbsTaskRetrieval):
    """Retrieval (typically reranking over ``top_ranked``) scored against float gains in the qrels.

    Attributes:
        gain_column: Name of the float-gain column in the qrels config.
        restrict_corpus_to_top_ranked: Encode only the documents that appear in ``top_ranked``.
            The float gains cover exactly those documents, and without this a bi-encoder would
            embed the whole corpus to rerank a handful of candidates per query.
    """

    gain_column: str = "gain"
    restrict_corpus_to_top_ranked: bool = True

    def dataset_transform(self, num_proc: int | None = None, **kwargs: Any) -> None:
        """Load the float gains for every (subset, split) and optionally trim the corpus."""
        self._float_gains: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
        for hf_subset, splits in self.dataset.items():
            self._float_gains[hf_subset] = {}
            for split, data in splits.items():
                self._float_gains[hf_subset][split] = self._load_float_gains(
                    hf_subset, split, num_proc
                )
                top_ranked = data.get("top_ranked")
                if self.restrict_corpus_to_top_ranked and top_ranked:
                    keep = {doc_id for docs in top_ranked.values() for doc_id in docs}
                    corpus = data["corpus"]
                    data["corpus"] = corpus.select(
                        [i for i, doc_id in enumerate(corpus["id"]) if doc_id in keep]
                    )

    def _load_float_gains(
        self, hf_subset: str, split: str, num_proc: int | None
    ) -> dict[str, dict[str, float]]:
        config = f"{hf_subset}-qrels" if hf_subset != "default" else "qrels"
        qrels = load_dataset(
            self.metadata.dataset["path"],
            config,
            split=split,
            revision=self.metadata.dataset["revision"],
            num_proc=num_proc,
        ).select_columns(["query-id", "corpus-id", self.gain_column])
        gains: dict[str, dict[str, float]] = defaultdict(dict)
        for query_id, doc_id, gain in zip(
            qrels["query-id"], qrels["corpus-id"], qrels[self.gain_column], strict=True
        ):
            if gain is not None:
                gains[str(query_id)][str(doc_id)] = float(gain)
        return dict(gains)

    def task_specific_scores(
        self,
        scores: dict[str, dict[str, float]],
        qrels: RelevantDocumentsType,
        results: dict[str, dict[str, float]],
        hf_split: str,
        hf_subset: str,
    ) -> dict[str, float]:
        """Adds ``ndcg_float_at_k`` over the float gains, for the queries the qrels metrics score."""
        gains = self._float_gains[hf_subset][hf_split]
        # exactly the queries the integer metrics average over: pytrec_eval scores the queries
        # present in `results` (an empty result dict scores 0; an absent query is skipped)
        scored = {
            query_id: results[query_id] for query_id in qrels if query_id in results
        }
        return ndcg_float_scores(gains, scored, self.k_values)
