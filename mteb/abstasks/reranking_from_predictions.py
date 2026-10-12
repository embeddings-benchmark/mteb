"""Reranking tasks with a fixed first-stage collection evaluated as subsets."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mteb.abstasks.first_stage_predictions import load_first_stage_predictions
from mteb.abstasks.retrieval import AbsTaskRetrieval

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from mteb.abstasks.first_stage_predictions import FirstStagePredictionSource
    from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData
    from mteb.models import MTEBModels
    from mteb.timing import TimingStack
    from mteb.types import ScoresDict


class AbsTaskRerankingFromPredictions(AbsTaskRetrieval):
    """Reuse a retrieval dataset with fixed candidates from multiple first stages.

    Declare ``retrieval_task``, ``first_stage_predictions`` and ``top_k`` on the
    task class. Metadata subsets are named ``{first_stage}__top{top_k}__{retrieval_subset}``,
    with the language mapping from the retrieval task. All subsets run by default;
    the usual language and ``hf_subsets`` filters select a smaller evaluation.

    Candidate sources and depth are part of the task definition. Changing them
    requires a new task version, just like changing the queries or relevance labels.
    """

    retrieval_task: type[AbsTaskRetrieval]
    first_stage_predictions: Mapping[str, str | Path | FirstStagePredictionSource]
    top_k: int
    k_values = (1, 3, 5, 10, 20, 50, 100, 1000)

    def load_data(
        self,
        num_proc: int | None = None,
        *,
        timer: TimingStack | None = None,
        **kwargs: Any,
    ) -> None:
        """Load shared retrieval data and the selected fixed candidate lists."""
        if self.data_loaded:
            return
        if self.top_k <= 0:
            raise ValueError("top_k must be positive.")
        subsets = {
            f"{first_stage}__top{self.top_k}__{subset}": (first_stage, subset)
            for first_stage in self.first_stage_predictions
            for subset in self.retrieval_task.metadata.hf_subsets
        }
        selected = {name: subsets[name] for name in self.hf_subsets}
        retrieval = self.retrieval_task(seed=self.seed)
        retrieval.filter_languages(
            languages=None,
            hf_subsets=list(dict.fromkeys(subset for _, subset in selected.values())),
        )
        retrieval.filter_eval_splits(self.eval_splits)
        retrieval.load_data(num_proc=num_proc, timer=timer, **kwargs)

        dataset: dict[str, dict[str, RetrievalSplitData]] = {}
        model_metas = {}
        for first_stage, source in self.first_stage_predictions.items():
            source_subsets = {
                name: subset
                for name, (stage, subset) in selected.items()
                if stage == first_stage
            }
            if not source_subsets:
                continue
            predictions = load_first_stage_predictions(
                source, retrieval.prediction_file_name
            )
            for name, subset in source_subsets.items():
                model_metas[name] = predictions["mteb_model_meta"]
                dataset[name] = {}
                for split in self.eval_splits:
                    top_ranked = {
                        query: sorted(scores, key=scores.__getitem__, reverse=True)[
                            : self.top_k
                        ]
                        for query, scores in predictions[subset][split].items()
                    }
                    # Share the corpus, queries and qrels, but keep candidate lists
                    # and the mutable split dictionary separate for each condition.
                    dataset[name][split] = {
                        **retrieval.dataset[subset][split],
                        "top_ranked": top_ranked,
                    }
        self.dataset = dataset
        self._first_stage_model_metas = model_metas
        self._top_k = self.top_k
        self.data_loaded = True

    def _evaluate_subset(
        self,
        model: MTEBModels,
        data_split: RetrievalSplitData,
        *,
        hf_subset: str,
        **kwargs: Any,
    ) -> ScoresDict:
        scores = super()._evaluate_subset(
            model, data_split, hf_subset=hf_subset, **kwargs
        )
        scores["previous_results_model_meta"] = self._first_stage_model_metas[hf_subset]
        scores["reranking_top_k"] = self._top_k
        return scores
