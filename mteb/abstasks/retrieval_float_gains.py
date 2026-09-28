"""Retrieval tasks whose qrels carry continuous relevance gains next to the integer labels.

The qrels config of such a task has two extra columns, ``gain`` (float) and ``theta``. The
standard loader reads only ``query-id``, ``corpus-id`` and ``score``, so every existing metric is
unchanged. This module adds the metric ``ndcg_float_at_k`` (NDCG over the float gains, see
:func:`ndcg_float_scores`) and a task class that reports it through
:meth:`AbsTaskRetrieval.task_specific_scores`. Nothing outside this module computes it.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from itertools import groupby
from typing import TYPE_CHECKING, Any

from datasets import get_dataset_config_names, get_dataset_split_names, load_dataset

from mteb._evaluators.retrieval_metrics import evaluate_abstention
from mteb.abstasks.retrieval import AbsTaskRetrieval

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData
    from mteb.types import RelevantDocumentsType

logger = logging.getLogger(__name__)


def _dcg(gains: Sequence[float], k: int) -> float:
    """DCG@k = sum_{r=1}^{k} gain(r) / log2(r + 1)."""
    return sum(
        gain / math.log2(rank + 1) for rank, gain in enumerate(gains[:k], start=1)
    )


def ndcg_float_scores(
    gains: Mapping[str, Mapping[str, float]],
    results: Mapping[str, Mapping[str, float]],
    k_values: Sequence[int],
) -> dict[str, float]:
    """Computes NDCG@k over continuous (float) relevance gains, bypassing pytrec_eval.

    pytrec_eval only accepts integer relevance labels. Here each document's gain is
    its float value, used as-is (linear gain, no ``2**g - 1``); gains must be finite
    and non-negative. Equal model scores form one equivalence class: every tied
    document is credited the group-mean gain, the expectation over all tie
    resolutions. A stable sort instead would let the candidate-pool order (which
    is relevance-ordered for reranking pools) leak ground truth into tied scores.

    The task layer must ensure that the gains cover every query in `results`;
    a query ID missing from `gains` raises `KeyError`. A non-finite or negative
    gain, and a NaN model score, raise `ValueError` rather than being scored --
    each would otherwise reach the mean as a silent `nan` or an extra tie class.
    Unlike the integer-qrels metrics, `skip_first_result` is not applied to the
    float metric.

    Args:
        gains: Continuous gains for each query, `{query_id: {doc_id: gain}}`. Must
            cover every query ID in `results`.
        results: Retrieval scores for each query, `{query_id: {doc_id: score}}`.
        k_values: The k values for which to compute the scores.

    Returns:
        A dictionary with the mean `ndcg_float_at_{k}` scores and the nAUC
        variants of the per-query scores.
    """
    for query_id, doc_gains in gains.items():
        # NaN passes every comparison, so finiteness is checked before the sign
        if any(not math.isfinite(gain) or gain < 0 for gain in doc_gains.values()):
            raise ValueError(
                f"Non-finite or negative gain for query {query_id}. Gains must be "
                "finite and non-negative."
            )

    for query_id, doc_scores in results.items():
        if any(math.isnan(score) for score in doc_scores.values()):
            raise ValueError(
                f"NaN model score for query {query_id}. NDCG_float is undefined "
                "for NaN model scores (infinities are ranked as usual)."
            )

    per_query: dict[str, list[float]] = defaultdict(list)
    for query_id, doc_scores in results.items():
        query_gains = gains[query_id]
        ranking = sorted(
            doc_scores, key=lambda doc_id: doc_scores[doc_id], reverse=True
        )

        tie_mean: dict[str, float] = {}
        for _, tie_group in groupby(ranking, key=doc_scores.__getitem__):
            tie_docs = list(tie_group)
            mean = sum(query_gains.get(doc_id, 0.0) for doc_id in tie_docs) / len(
                tie_docs
            )
            tie_mean.update(dict.fromkeys(tie_docs, mean))

        ideal_gains = sorted(query_gains.values(), reverse=True)
        for k in k_values:
            ideal_dcg = _dcg(ideal_gains, k)
            if ideal_dcg == 0.0:
                per_query[f"NDCG_float@{k}"].append(0.0)
                continue
            actual_dcg = _dcg([tie_mean[doc_id] for doc_id in ranking[:k]], k)
            per_query[f"NDCG_float@{k}"].append(actual_dcg / ideal_dcg)

    summary = {
        f"ndcg_float_at_{key.split('@')[1]}": round(sum(values) / len(values), 5)
        for key, values in per_query.items()
    }
    naucs = evaluate_abstention(results, per_query)
    return {
        **summary,
        **{key.replace("@", "_at_").lower(): value for key, value in naucs.items()},
    }


class AbsTaskRetrievalFloatGains(AbsTaskRetrieval):
    """Retrieval (typically reranking over ``top_ranked``) scored against float gains in the qrels.

    The gains are read from the qrels config of each subset (``{subset}-qrels``, or ``default`` /
    ``qrels`` for the default subset), resolved like the standard retrieval loader resolves it.
    ``ignore_identical_ids`` is honoured: each query's own document is dropped from the ranking and
    from the gains, so it cannot inflate the ideal DCG. ``skip_first_result`` is not supported and
    raises a ``ValueError``.

    Two ways to evaluate:

    - Reranking (default, ``rerank_top_ranked = True``): each query is scored over its ``top_ranked``
      candidates, the documents that carry a gain; ``ndcg_float_at_k`` is reported next to the
      integer-qrels metrics.
    - Full-corpus retrieval (``task.as_full_corpus_retrieval()``, which sets ``rerank_top_ranked =
      False`` under its own task name): ``top_ranked`` is dropped, so each
      query searches the whole corpus, minus the documents listed in the optional
      ``{subset}-excluded`` config (``query-id``, ``excluded-corpus-ids``). Gains exist only for the
      candidate pools, so only the integer-qrels metrics are reported, and a float-gain main score
      falls back to ``ndcg_at_10``.

    Attributes:
        gain_column: Name of the float-gain column in the qrels config.
        rerank_top_ranked: Rerank ``top_ranked`` (``True``) or retrieve from the full corpus.
        restrict_corpus_to_top_ranked: When reranking, encode only the documents that appear in
            ``top_ranked``. The float gains cover exactly those documents, and without this a
            bi-encoder would embed the whole corpus to rerank a handful of candidates per query.
            Descriptive statistics describe the full corpus: set this to ``False`` before calling
            ``calculate_descriptive_statistics``.
    """

    gain_column: str = "gain"
    rerank_top_ranked: bool = True
    restrict_corpus_to_top_ranked: bool = True

    def dataset_transform(self, num_proc: int | None = None, **kwargs: Any) -> None:
        """Load the float gains for every (subset, split) and optionally trim the corpus."""
        self._float_gains: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
        for hf_subset, splits in self.dataset.items():
            self._float_gains[hf_subset] = {}
            for split, data in splits.items():
                if not self.rerank_top_ranked:
                    data["top_ranked"] = self._full_corpus_candidates(
                        hf_subset, split, data, num_proc
                    )
                    continue
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

    def _full_corpus_candidates(
        self,
        hf_subset: str,
        split: str,
        data: RetrievalSplitData,
        num_proc: int | None,
    ) -> dict[str, list[str]] | None:
        """``None`` (plain full-corpus search), or the full corpus minus each query's excluded ids."""
        path = self.metadata.dataset["path"]
        revision = self.metadata.dataset["revision"]
        config = f"{hf_subset}-excluded" if hf_subset != "default" else "excluded"
        if config not in get_dataset_config_names(path, revision):
            return None
        _, excluded_split = self._split_of(config, split)
        rows = load_dataset(
            path, config, split=excluded_split, revision=revision, num_proc=num_proc
        )
        excluded = {
            str(q): set(map(str, ids))
            for q, ids in zip(
                rows["query-id"], rows["excluded-corpus-ids"], strict=True
            )
        }
        corpus_ids = [str(d) for d in data["corpus"]["id"]]
        return {
            str(q): [d for d in corpus_ids if d not in excluded[str(q)]]
            if str(q) in excluded
            else corpus_ids
            for q in data["queries"]["id"]
        }

    def _split_of(self, config: str, split: str) -> tuple[str, str]:
        """The split of ``config`` to read: ``split``, or the only split (like the core loader)."""
        splits = get_dataset_split_names(
            self.metadata.dataset["path"],
            revision=self.metadata.dataset["revision"],
            config_name=config,
        )
        if split not in splits:
            if len(splits) != 1:
                raise ValueError(
                    f"Split {split} not found in {splits}. Please specify a valid split."
                )
            split = str(splits[0])
        return config, split

    def _qrels_config_and_split(self, hf_subset: str, split: str) -> tuple[str, str]:
        """The qrels config and split, resolved like ``RetrievalDatasetLoader``."""
        if hf_subset != "default":
            config = f"{hf_subset}-qrels"
        else:
            configs = get_dataset_config_names(
                self.metadata.dataset["path"], self.metadata.dataset["revision"]
            )
            config = "default" if "default" in configs else "qrels"
        return self._split_of(config, split)

    def _load_float_gains(
        self, hf_subset: str, split: str, num_proc: int | None
    ) -> dict[str, dict[str, float]]:
        config, qrels_split = self._qrels_config_and_split(hf_subset, split)
        qrels = load_dataset(
            self.metadata.dataset["path"],
            config,
            split=qrels_split,
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

    def _gains_for(self, hf_subset: str, hf_split: str) -> dict[str, dict[str, float]]:
        """The gains of one (subset, split), loaded on first use if ``dataset_transform`` did not run."""
        if not hasattr(self, "_float_gains"):
            self._float_gains = {}
        subset_gains = self._float_gains.setdefault(hf_subset, {})
        if hf_split not in subset_gains:
            subset_gains[hf_split] = self._load_float_gains(hf_subset, hf_split, None)
        return subset_gains[hf_split]

    def as_full_corpus_retrieval(self) -> AbsTaskRetrievalFloatGains:
        """A full-corpus retrieval version of this task (``rerank_top_ranked = False``).

        It gets its own name (``<name>.retrieval``) and ``main_score = "ndcg_at_10"``, so its
        results are not stored under, and do not overwrite, the reranking task's results.
        """
        task = type(self)()
        task.rerank_top_ranked = False
        task.metadata = self.metadata.model_copy(
            update={
                "name": f"{self.metadata.name}.retrieval",
                "main_score": "ndcg_at_10",
                "description": f"{self.metadata.description} Full-corpus retrieval view: "
                "only the integer-qrels metrics are reported.",
            }
        )
        return task

    def _add_main_score(self, scores: dict[str, Any]) -> None:
        # the full-corpus mode reports only the integer-qrels metrics, so a main score over
        # the float gains falls back to the standard NDCG@10
        main_score = self.metadata.main_score
        if not self.rerank_top_ranked and main_score.startswith("ndcg_float"):
            main_score = "ndcg_at_10"
        scores["main_score"] = scores[main_score]

    def task_specific_scores(
        self,
        scores: dict[str, dict[str, float]],
        qrels: RelevantDocumentsType,
        results: dict[str, dict[str, float]],
        hf_split: str,
        hf_subset: str,
    ) -> dict[str, float]:
        """Adds ``ndcg_float_at_k`` over the float gains, for the queries the qrels metrics score."""
        if not self.rerank_top_ranked:
            return {}
        if self.skip_first_result:
            raise ValueError(
                "skip_first_result is not supported by the float-gains metric."
            )
        gains = self._gains_for(hf_subset, hf_split)
        # exactly the queries the integer metrics average over: pytrec_eval scores the queries
        # present in `results` (an empty result dict scores 0; an absent query is skipped)
        scored = {
            query_id: results[query_id] for query_id in qrels if query_id in results
        }
        if self.ignore_identical_ids:
            # the evaluator already dropped each query's own document from `results`; drop it
            # from the gains too (on copies), so it cannot inflate the ideal DCG
            gains = {
                query_id: {d: g for d, g in docs.items() if d != query_id}
                for query_id, docs in gains.items()
            }
            scored = {
                query_id: {d: s for d, s in docs.items() if d != query_id}
                for query_id, docs in scored.items()
            }
        return ndcg_float_scores(gains, scored, self.k_values)
