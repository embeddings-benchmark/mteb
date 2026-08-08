from __future__ import annotations

import json
import logging
import warnings
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from datasets import Dataset, DatasetDict, concatenate_datasets

from mteb._create_dataloaders import (
    _combine_queries_with_instruction_text,
    _convert_conv_history_to_query,
    _corpus_to_dict,
)
from mteb._evaluators import RetrievalEvaluator
from mteb._evaluators.retrieval_metrics import make_score_dict
from mteb.models import (
    CrossEncoderProtocol,
    EncoderProtocol,
    SearchCrossEncoderWrapper,
    SearchEncoderWrapper,
    SearchProtocol,
)
from mteb.timing import TimingStack
from mteb.types import (
    PromptType,
)
from mteb.types.statistics import RetrievalDescriptiveStatistics

from ._data_filter.dataset_filters import (
    iter_texts,
    keep_first_occurrence,
    keep_long_enough,
    text_key,
)
from ._statistics_calculation import (
    calculate_relevant_docs_statistics,
    calculate_single_input_modality_statistics,
    calculate_top_ranked_statistics,
)
from .abstask import AbsTask, _no_split_matched_message
from .retrieval_dataset_loaders import (
    RetrievalDatasetLoader,
    _combine_queries_with_instructions_datasets,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from typing_extensions import Self

    from mteb.models import (
        MTEBModels,
    )
    from mteb.types import (
        EncodeKwargs,
        HFSubset,
        Modalities,
        QueryDatasetType,
        RelevantDocumentsType,
        RetrievalOutputType,
        ScoresDict,
    )

    from ._data_filter.dataset_filters import (
        KeepIndicesFn,
        TextLengthUnit,
        TextNormalization,
    )
    from .retrieval_dataset_loaders import (
        RetrievalSplitData,
    )

logger = logging.getLogger(__name__)


def _filter_queries_without_positives(
    relevant_docs: RelevantDocumentsType, queries: QueryDatasetType
) -> tuple[RelevantDocumentsType, QueryDatasetType]:
    _relevant_docs = {}
    for idx in relevant_docs:
        if len(relevant_docs[idx]) == 0:  # no relevant docs
            continue
        _relevant_docs[idx] = relevant_docs[idx]

    ids_to_keep = set(_relevant_docs.keys())
    indices = [i for i, id_ in enumerate(queries["id"]) if id_ in ids_to_keep]
    queries = queries.select(indices)

    return _relevant_docs, queries


def _select_kept_entries(
    dataset: Dataset,
    keep_fn: KeepIndicesFn,
    columns: Sequence[str],
    *,
    remap_duplicates: TextNormalization | None,
) -> tuple[Dataset, set[str], dict[str, str]]:
    """Apply `keep_fn` to a corpus or query dataset.

    Remapping assumes that `keep_fn` keeps the *first* entry of each group of equal texts, which lets the
    replacements be collected in a single pass: a removed entry always follows the entry it is remapped onto. It
    also has to compare texts the same way `keep_fn` does, which is why it takes the normalization rather than a
    flag.

    Returns:
        The filtered dataset, the ids it kept, and a mapping from the id of a removed entry to the id of the first
        kept entry with the same text. That mapping is empty unless `remap_duplicates` was given.
    """
    keep = keep_fn(iter_texts(dataset, columns))
    ids = dataset["id"]
    kept_ids = {ids[i] for i in keep}

    replacements: dict[str, str] = {}
    if remap_duplicates is not None:
        keep_set = set(keep)
        canonical: dict[bytes, str] = {}
        for i, row in enumerate(iter_texts(dataset, columns)):
            key = text_key(row, remap_duplicates)
            if i in keep_set:
                canonical.setdefault(key, ids[i])
            elif (target := canonical.get(key)) is not None:
                replacements[ids[i]] = target

    return dataset.select(keep), kept_ids, replacements


def _filter_retrieval_split(  # noqa: PLR0914
    split_data: RetrievalSplitData,
    keep_fn: KeepIndicesFn,
    columns: Sequence[str],
    *,
    remap_duplicates: TextNormalization | None,
) -> tuple[RetrievalSplitData, int]:
    """Apply `keep_fn` to the corpus and the queries of a single split, keeping the relevance judgements valid.

    Args:
        split_data: The corpus, queries, relevance judgements and top-ranked documents of one split.
        keep_fn: Decides which documents and queries to keep.
        columns: The text columns of the corpus and the queries to hand to `keep_fn`.
        remap_duplicates: The text normalization to use when handing the relevance judgements of a removed
            document or query over to the first kept entry with the same text. This is what makes deduplication
            lossless; for a filter that removes entries on their own merit, such as a length filter, it must be
            None.

    Returns:
        The filtered split and the number of documents and queries that were removed.

    Raises:
        ValueError: If one of `columns` is missing from the corpus or from the queries.
    """
    old_corpus, old_queries = split_data["corpus"], split_data["queries"]
    missing = [
        column
        for column in columns
        if column not in old_corpus.column_names
        or column not in old_queries.column_names
    ]
    if missing:
        raise ValueError(
            f"Cannot filter on {missing}: the corpus has the columns {old_corpus.column_names} and the queries "
            f"have the columns {old_queries.column_names}."
        )

    corpus, kept_doc_ids, doc_replacements = _select_kept_entries(
        old_corpus, keep_fn, columns, remap_duplicates=remap_duplicates
    )
    queries, kept_query_ids, query_replacements = _select_kept_entries(
        old_queries, keep_fn, columns, remap_duplicates=remap_duplicates
    )

    relevant_docs: dict[str, dict[str, int]] = {}
    for query_id, docs in split_data["relevant_docs"].items():
        query_id = query_replacements.get(query_id, query_id)  # noqa: PLW2901
        if query_id not in kept_query_ids:
            continue
        scores = relevant_docs.setdefault(query_id, {})
        for doc_id, score in docs.items():
            doc_id = doc_replacements.get(doc_id, doc_id)  # noqa: PLW2901
            if doc_id in kept_doc_ids:
                scores[doc_id] = max(scores.get(doc_id, score), score)

    relevant_docs, queries = _filter_queries_without_positives(  # type: ignore[assignment]
        relevant_docs, queries
    )

    top_ranked = split_data["top_ranked"]
    if top_ranked is not None:
        remaining_query_ids = set(queries["id"])
        new_top_ranked: dict[str, list[str]] = {}
        for query_id, doc_ids in top_ranked.items():
            query_id = query_replacements.get(query_id, query_id)  # noqa: PLW2901
            if query_id not in remaining_query_ids:
                continue
            ranked = new_top_ranked.setdefault(query_id, [])
            for doc_id in doc_ids:
                doc_id = doc_replacements.get(doc_id, doc_id)  # noqa: PLW2901
                if doc_id in kept_doc_ids and doc_id not in ranked:
                    ranked.append(doc_id)
        top_ranked = new_top_ranked

    n_removed = len(old_corpus) + len(old_queries) - len(corpus) - len(queries)
    return {
        "corpus": corpus,
        "queries": queries,
        "relevant_docs": relevant_docs,
        "top_ranked": top_ranked,
    }, n_removed


class AbsTaskRetrieval(AbsTask):
    """The class which retrieval tasks inherit from.

    A retrieval task consists of a corpus of documents, a set of queries, and a mapping of which documents are relevant for each query.
    The task is to retrieve the relevant documents for each query. The evaluation is done by indexing the corpus and then searching for each query.
    The retrieved documents are then compared to the relevant documents to calculate the evaluation scores.


    Attributes:
        dataset: A nested dictionary where the first key is the subset (language or "default"),
                 the second key is the split (e.g., "train", "test"), and the value is a RetrievalSplitData object.
        ignore_identical_ids: If True, identical IDs in queries and corpus are ignored during evaluation.
        k_values: A sequence of integers representing the k values for evaluation metrics.
        skip_first_result: If True, the first result is skipped during evaluation
        abstask_prompt: Prompt to use for the task for instruction model if not prompt is provided in TaskMetadata.prompt.
    """

    ignore_identical_ids: bool = False
    abstask_prompt = "Retrieve text based on user query."
    k_values: Sequence[int] = (1, 3, 5, 10, 20, 100, 1000)
    _top_k: int = max(k_values)
    dataset: dict[str, dict[str, RetrievalSplitData]]
    _support_cross_encoder: bool = True
    _support_search: bool = True
    _previous_results_model_meta: dict[str, Any] | None = None
    skip_first_result: bool = False

    def convert_v1_dataset_format_to_v2(
        self,
        num_proc: int | None,
    ) -> None:
        """Convert dataset from v1 (from `self.queries`, `self.document`) format to v2 format (`self.dotaset`)."""
        # check if dataset is `v1` version
        if (
            not hasattr(self, "queries")
            or not hasattr(self, "corpus")
            or not hasattr(self, "relevant_docs")
        ):
            return

        self.dataset = {}

        def _process_split(
            ds_queries: dict[str, Any] | Dataset, ds_corpus: dict[str, Any] | Dataset
        ) -> tuple[Dataset, Dataset]:
            if isinstance(ds_queries, dict):
                queries = Dataset.from_list(
                    [{"id": k, "text": v} for k, v in ds_queries.items()]
                )
            elif isinstance(ds_queries, Dataset):
                queries = ds_queries
            else:
                raise ValueError(f"Can't convert queries of type {type(ds_queries)}")

            if isinstance(ds_corpus, dict):
                corpus = Dataset.from_list(
                    [
                        {
                            "id": k,
                            "text": v if isinstance(v, str) else v["text"],
                            "title": v.get("title", "") if isinstance(v, dict) else "",
                        }
                        for k, v in ds_corpus.items()
                    ]
                )
            elif isinstance(ds_corpus, Dataset):
                corpus = ds_corpus
            else:
                raise ValueError(f"Can't convert corpus of type {type(ds_corpus)}")
            return queries, corpus

        if self.metadata.is_multilingual:
            for subset in self.queries:
                if subset not in self.dataset:
                    self.dataset[subset] = {}
                for split in self.queries[subset]:
                    if split not in self.dataset[subset]:
                        self.dataset[subset][split] = {}  # type: ignore[typeddict-item]
                    queries = self.queries[subset][split]
                    corpus = self.corpus[subset][split]

                    (
                        self.dataset[subset][split]["queries"],
                        self.dataset[subset][split]["corpus"],
                    ) = _process_split(queries, corpus)

                    self.dataset[subset][split]["relevant_docs"] = self.relevant_docs[
                        subset
                    ][split]
                    if hasattr(self, "instructions"):
                        instructions = self.instructions[subset][split]
                        self.dataset[subset][split]["queries"] = (
                            _combine_queries_with_instructions_datasets(
                                self.dataset[subset][split]["queries"],
                                instructions,
                                num_proc,
                            )
                        )
                    if hasattr(self, "top_ranked"):
                        self.dataset[subset][split]["top_ranked"] = self.top_ranked[
                            subset
                        ][split]
                    else:
                        self.dataset[subset][split]["top_ranked"] = None
        else:
            subset = "default"
            if subset not in self.dataset:
                self.dataset[subset] = {}
            for split in self.queries:
                if split not in self.dataset[subset]:
                    self.dataset[subset][split] = {}  # type: ignore[typeddict-item]
                queries = self.queries[split]
                corpus = self.corpus[split]
                (
                    self.dataset[subset][split]["queries"],
                    self.dataset[subset][split]["corpus"],
                ) = _process_split(queries, corpus)

                self.dataset[subset][split]["relevant_docs"] = self.relevant_docs[
                    split
                ].copy()
                if hasattr(self, "instructions"):
                    instructions = self.instructions[split]
                    self.dataset[subset][split]["queries"] = (
                        _combine_queries_with_instructions_datasets(
                            self.dataset[subset][split]["queries"],
                            instructions,
                            num_proc,
                        )
                    )
                if hasattr(self, "top_ranked") and self.top_ranked:
                    self.dataset[subset][split]["top_ranked"] = self.top_ranked[
                        split
                    ].copy()
                else:
                    self.dataset[subset][split]["top_ranked"] = None

        del self.queries
        del self.corpus
        del self.relevant_docs
        if hasattr(self, "instructions"):
            del self.instructions
        if hasattr(self, "top_ranked"):
            del self.top_ranked

    def load_data(
        self,
        num_proc: int | None = None,
        *,
        timer: TimingStack | None = None,
        **kwargs: Any,
    ) -> None:
        """Load the dataset for the retrieval task."""
        if self.data_loaded:
            return

        self.dataset = {}
        dataset_path = self.metadata.dataset["path"]
        eval_splits = self.eval_splits
        trust_remote_code = self.metadata.dataset.get("trust_remote_code", False)
        revision = self.metadata.dataset["revision"]

        def _process_data(split: str, hf_subset: str = "default") -> None:
            """Helper function to load and process data for a given split and language"""
            logger.debug(
                f"Loading {split} split for {hf_subset} subset of {self.metadata.name}"
            )
            if hf_subset not in self.dataset:
                self.dataset[hf_subset] = {}

            self.dataset[hf_subset][split] = RetrievalDatasetLoader(
                hf_repo=dataset_path,
                revision=revision,
                trust_remote_code=trust_remote_code,
                split=split,
                config=hf_subset,
            ).load(
                num_proc=num_proc,
            )

        timer = timer or TimingStack()
        with timer(
            "Data loading", log_message=f"Loading dataset {self.metadata.name}..."
        ):
            if self.metadata.is_multilingual:
                for lang in self.hf_subsets:
                    for split in eval_splits:
                        _process_data(split, lang)
            else:
                for split in eval_splits:
                    _process_data(split)

        with timer("Dataset transform"):
            self.dataset_transform(num_proc=num_proc)
        self.data_loaded = True

    def _get_text_columns(self) -> list[str]:  # noqa: PLR6301
        return ["text"]

    def _warn_about_unusable_data(self) -> None:
        """Warn about splits that a filter left without documents or without queries."""
        for subset, splits_data in self.dataset.items():
            for split, split_data in splits_data.items():
                empty = sorted(
                    name
                    for name in ("corpus", "queries")
                    if len(split_data[name]) == 0  # type: ignore[literal-required]
                )
                if empty:
                    msg = (
                        f"Filtering left the {' and the '.join(empty)} of the '{split}' split of "
                        f"'{self.metadata.name}' (subset '{subset}') empty. Evaluating it will fail."
                    )
                    logger.warning(msg)
                    warnings.warn(msg, stacklevel=2)

    def _filter_retrieval(
        self,
        keep_fn: KeepIndicesFn,
        *,
        filter_name: str,
        remap_duplicates: TextNormalization | None,
        columns: Sequence[str] | None,
        splits: Sequence[str] | None,
        subsets: Sequence[HFSubset] | None,
        num_proc: int | None,
    ) -> Self:
        if not self.data_loaded:
            self.load_data(num_proc=num_proc)
        if self.dataset is None:
            raise ValueError(f"Dataset of task '{self.metadata.name}' is not loaded.")

        text_columns = (
            list(columns) if columns is not None else self._get_text_columns()
        )
        if not text_columns:
            raise NotImplementedError(
                f"`{filter_name}` does not know which columns of '{self.metadata.name}' hold text. This is "
                "expected for tasks without a text modality; pass `columns=[...]` to filter on specific columns."
            )

        n_removed = 0
        n_filtered_splits = 0
        for subset, splits_data in self.dataset.items():
            if subsets is not None and subset not in subsets:
                continue
            for split in list(splits_data.keys()):
                if splits is not None and split not in splits:
                    continue
                splits_data[split], removed = _filter_retrieval_split(
                    splits_data[split],
                    keep_fn,
                    text_columns,
                    remap_duplicates=remap_duplicates,
                )
                n_removed += removed
                n_filtered_splits += 1

        if n_filtered_splits == 0:
            raise ValueError(
                _no_split_matched_message(self.metadata.name, self.dataset)
            )

        if n_removed:
            self._mark_data_modified()
            self._warn_about_unusable_data()

        logger.info(
            f"`{filter_name}` removed {n_removed} documents and queries from '{self.metadata.name}' "
            f"(columns={text_columns})."
        )
        return self

    def remove_duplicates(
        self,
        *,
        normalize: TextNormalization = "strip",
        columns: Sequence[str] | None = None,
        splits: Sequence[str] | None = None,
        subsets: Sequence[HFSubset] | None = None,
        num_proc: int | None = None,
    ) -> Self:
        """Remove duplicated documents and queries from the task, keeping the first occurrence of each.

        Relevance judgements that point at a removed duplicate are moved to the copy that was kept, so no query
        loses a positive document, and duplicated queries inherit the union of their relevance judgements. Any
        query left without a positive document afterwards is dropped, as it cannot be scored.

        The data is loaded if it has not been loaded yet, and the task's dataset is modified in place. Because the
        dataset then no longer matches the published one, scores computed after cleaning are not comparable to the
        results on the leaderboard.

        Args:
            normalize: How much of a difference between two texts to ignore when comparing them: `"strip"` (the
                default) only ignores surrounding whitespace, `"casefold"` also ignores case, and `"alphanumeric"`
                also ignores punctuation and repeated whitespace. The looser settings catch more duplicates but can
                merge documents that a reader would tell apart, and case folding is not meaningful in every script.
            columns: The columns of the corpus and the queries to compare. Defaults to `["text"]`.
            splits: The splits to filter. Defaults to every loaded split.
            subsets: The Huggingface subsets to filter. Defaults to every loaded subset.
            num_proc: Number of processes to use for loading the dataset.

        Returns:
            The task itself, so that calls can be chained.

        Raises:
            ValueError: If `splits` or `subsets` select none of the task's data.

        Examples:
            >>> import mteb
            >>> task = mteb.get_task("SciFact")
            >>> task.remove_duplicates()
        """
        return self._filter_retrieval(
            keep_first_occurrence(normalize),
            filter_name="remove_duplicates",
            remap_duplicates=normalize,
            columns=columns,
            splits=splits,
            subsets=subsets,
            num_proc=num_proc,
        )

    def filter_short_documents(
        self,
        min_length: int = 5,
        *,
        unit: TextLengthUnit = "characters",
        columns: Sequence[str] | None = None,
        splits: Sequence[str] | None = None,
        subsets: Sequence[HFSubset] | None = None,
        num_proc: int | None = None,
    ) -> Self:
        """Remove documents and queries shorter than `min_length` from the task.

        Relevance judgements referring to a removed document or query are dropped along with it, and any query
        left without a positive document afterwards is dropped as well, as it cannot be scored.

        The data is loaded if it has not been loaded yet, and the task's dataset is modified in place. Because the
        dataset then no longer matches the published one, scores computed after cleaning are not comparable to the
        results on the leaderboard.

        Args:
            min_length: The minimum length a document or query must have to be kept.
            unit: Whether `min_length` counts `"characters"` or whitespace-separated `"words"`. Word counts are a
                poor fit for languages that are not whitespace-delimited, such as Chinese or Japanese.
            columns: The columns of the corpus and the queries to measure. Defaults to `["text"]`.
            splits: The splits to filter. Defaults to every loaded split.
            subsets: The Huggingface subsets to filter. Defaults to every loaded subset.
            num_proc: Number of processes to use for loading the dataset.

        Returns:
            The task itself, so that calls can be chained.

        Raises:
            ValueError: If `splits` or `subsets` select none of the task's data.

        Examples:
            >>> import mteb
            >>> task = mteb.get_task("SciFact")
            >>> task.filter_short_documents(min_length=5)
        """
        return self._filter_retrieval(
            keep_long_enough(min_length, unit),
            filter_name="filter_short_documents",
            remap_duplicates=None,
            columns=columns,
            splits=splits,
            subsets=subsets,
            num_proc=num_proc,
        )

    def evaluate(
        self,
        model: MTEBModels,
        split: str = "test",
        subsets_to_run: list[HFSubset] | None = None,
        *,
        encode_kwargs: EncodeKwargs,
        prediction_folder: Path | None = None,
        num_proc: int | None = None,
        timer: TimingStack | None = None,
        **kwargs: Any,
    ) -> Mapping[HFSubset, ScoresDict]:
        """Evaluate the model on the retrieval task.

        Args:
            model: Model to evaluate. Model should implement the [SearchProtocol][mteb.models.models_protocols.SearchProtocol]
                or be an [Encoder][mteb.models.models_protocols.EncoderProtocol] or [CrossEncoderProtocol][mteb.models.models_protocols.CrossEncoderProtocol].
            split: Split to evaluate on
            subsets_to_run: Optional list of subsets to evaluate on
            encode_kwargs: Keyword arguments passed to the encoder
            prediction_folder: Folder to save model predictions
            num_proc: Number of processes to use
            timer: A context manager that tracks the timing of evaluation phases.
            **kwargs: Additional keyword arguments passed to the evaluator

        Returns:
            Dictionary mapping subsets to their evaluation scores
        """
        timer = timer or TimingStack()
        if not self.data_loaded:
            self.load_data(num_proc=num_proc, timer=timer)
        # TODO: convert all tasks directly https://github.com/embeddings-benchmark/mteb/issues/2030
        self.convert_v1_dataset_format_to_v2(num_proc=num_proc)

        return super().evaluate(
            model,
            split,
            subsets_to_run,
            encode_kwargs=encode_kwargs,
            prediction_folder=prediction_folder,
            num_proc=num_proc,
            timer=timer,
            **kwargs,
        )

    def _evaluate_subset(
        self,
        model: MTEBModels,
        data_split: RetrievalSplitData,
        *,
        encode_kwargs: EncodeKwargs,
        hf_split: str,
        hf_subset: str,
        prediction_folder: Path | None = None,
        num_proc: int | None = None,
        timer: TimingStack,
        **kwargs: Any,
    ) -> ScoresDict:
        """Evaluate a model on a specific subset of the data.

        Args:
            model: Model to evaluate
            data_split: Data split to evaluate on
            encode_kwargs: Keyword arguments passed to the encoder
            hf_split: Split to evaluate on
            hf_subset: Subset to evaluate on
            prediction_folder: Folder with results prediction
            num_proc: Number of processes to use
            timer: A context manager that tracks the timing of evaluation phases.
            **kwargs: Additional keyword arguments passed to the evaluator

        Returns:
            Dictionary of evaluation scores
        """
        # ensure queries format (see #3030)
        data_split["relevant_docs"], data_split["queries"] = (
            _filter_queries_without_positives(
                data_split["relevant_docs"], data_split["queries"]
            )
        )
        retriever = RetrievalEvaluator(
            corpus=data_split["corpus"],
            queries=data_split["queries"],
            task_metadata=self.metadata,
            hf_split=hf_split,
            hf_subset=hf_subset,
            top_ranked=data_split["top_ranked"],
            top_k=self._top_k,
            timer=timer,
            **kwargs,
        )

        search_model: SearchProtocol

        if isinstance(model, EncoderProtocol) and not isinstance(model, SearchProtocol):
            search_model = SearchEncoderWrapper(model)
        elif isinstance(model, CrossEncoderProtocol):
            search_model = SearchCrossEncoderWrapper(model)
        elif isinstance(model, SearchProtocol):
            search_model = model
        else:
            raise TypeError(
                f"RetrievalEvaluator expects a SearchInterface, Encoder, or CrossEncoder, got {type(model)}"
            )

        results = retriever(
            search_model,
            encode_kwargs=encode_kwargs,
            num_proc=num_proc,
        )

        if prediction_folder:
            self._save_task_predictions(
                results,
                model,
                prediction_folder,
                hf_subset=hf_subset,
                hf_split=hf_split,
            )

        with timer(
            "Scoring",
            split=hf_split,
            subset=hf_subset,
            log_message="Running retrieval task - Evaluating retrieval scores...",
        ):
            (
                all_scores,
                ndcg,
                _map,
                recall,
                precision,
                naucs,
                mrr,
                naucs_mrr,
                hit_rate,
            ) = retriever.evaluate(
                data_split["relevant_docs"],
                results,
                self.k_values,
                ignore_identical_ids=self.ignore_identical_ids,
                skip_first_result=self.skip_first_result,
            )

        task_specific_scores = self.task_specific_scores(
            all_scores,
            data_split["relevant_docs"],
            results,
            hf_split=hf_split,
            hf_subset=hf_subset,
        )
        logger.info("Running retrieval task - Finished.")
        return make_score_dict(
            ndcg=ndcg,
            _map=_map,
            recall=recall,
            precision=precision,
            mrr=mrr,
            naucs=naucs,
            naucs_mrr=naucs_mrr,
            hit_rate=hit_rate,
            task_scores=task_specific_scores,
            previous_results_model_meta=self._previous_results_model_meta,
        )

    def task_specific_scores(  # noqa: PLR6301
        self,
        scores: dict[str, dict[str, float]],
        qrels: RelevantDocumentsType,
        results: dict[str, dict[str, float]],
        hf_split: str,
        hf_subset: str,
    ) -> dict[str, float]:
        """Calculate task specific scores. Override in subclass if needed.

        Args:
            scores: Dictionary of scores
            qrels: Relevant documents
            results: Retrieval results
            hf_split: Split to evaluate on
            hf_subset: Subset to evaluate on
        """
        return {}

    def _calculate_descriptive_statistics_from_split(  # noqa: PLR0914
        self,
        split: str,
        *,
        hf_subset: str | None = None,
        compute_overall: bool = False,
        num_proc: int | None = None,
    ) -> RetrievalDescriptiveStatistics:
        self.convert_v1_dataset_format_to_v2(num_proc)
        if hf_subset and hf_subset in self.dataset:
            split_data = self.dataset[hf_subset][split]
            queries = split_data["queries"]
            corpus = split_data["corpus"]
            relevant_docs = split_data["relevant_docs"]
            top_ranked = split_data["top_ranked"]
        elif compute_overall:
            queries = None
            corpus = None
            relevant_docs = {}
            top_ranked = {}
            for hf_subset in self.metadata.eval_langs:  # noqa: PLR1704
                split_data = self.dataset[hf_subset][split]
                if queries is None:
                    queries = split_data["queries"]
                else:
                    queries = concatenate_datasets([queries, split_data["queries"]])
                if corpus is None:
                    corpus = split_data["corpus"]
                else:
                    corpus = concatenate_datasets([corpus, split_data["corpus"]])

                relevant_docs.update(
                    _process_relevant_docs(
                        split_data["relevant_docs"], hf_subset, split
                    )
                )

                if "top_ranked" in split_data and split_data["top_ranked"] is not None:
                    top_ranked.update(
                        {
                            f"{split}_{hf_subset}_{k}": v
                            for k, v in split_data["top_ranked"].items()
                        }
                    )
        else:
            if "default" in self.dataset and split != "default":
                return self._calculate_descriptive_statistics_from_split(
                    split=split, hf_subset="default"
                )
            split_data = self.dataset["default"][split]
            queries = split_data["queries"]
            corpus = split_data["corpus"]
            relevant_docs = split_data["relevant_docs"]
            top_ranked = split_data["top_ranked"]

        num_documents = len(corpus)
        num_queries = len(queries)

        if self.metadata.category is None:
            queries_modalities: Sequence[str] = ["text"]
            corpus_modalities: Sequence[str] = ["text"]
        else:
            queries_modalities = self.metadata.get_modalities(
                prompt_type=PromptType.query
            )
            corpus_modalities = self.metadata.get_modalities(
                prompt_type=PromptType.document
            )

        # Build corpus col_inputs — text needs special mapping from the corpus dict format.
        corpus_col_inputs: dict[Modalities, list[Any]] = {}
        if "text" in corpus_modalities:
            corpus_col_inputs["text"] = corpus.map(_corpus_to_dict)["text"]
        if "image" in corpus_modalities:
            corpus_col_inputs["image"] = corpus["image"]
        if "audio" in corpus_modalities:
            corpus_col_inputs["audio"] = corpus["audio"]
        if "video" in corpus_modalities:
            corpus_col_inputs["video"] = corpus["video"]

        # Build queries col_inputs — text may need instruction/conversation transformations.
        queries_col_inputs: dict[Modalities, list[Any]] = {}
        if "text" in queries_modalities:
            queries_ = queries
            if "instruction" in queries_[0]:
                queries_ = _combine_queries_with_instruction_text(queries_)
            if isinstance(queries_["text"][0], dict | list):
                queries_ = queries_.map(_convert_conv_history_to_query)
            queries_col_inputs["text"] = queries_["text"]
        if "image" in queries_modalities:
            queries_col_inputs["image"] = queries["image"]
        if "audio" in queries_modalities:
            queries_col_inputs["audio"] = queries["audio"]
        if "video" in queries_modalities:
            queries_col_inputs["video"] = queries["video"]

        corpus_stats = calculate_single_input_modality_statistics(
            corpus_col_inputs, max_workers=num_proc
        )
        queries_stats = calculate_single_input_modality_statistics(
            queries_col_inputs, max_workers=num_proc
        )

        number_of_characters = sum(
            stat["total_text_length"]
            for stat in [
                corpus_stats["text_statistics"],
                queries_stats["text_statistics"],
            ]
            if stat is not None
        )

        relevant_docs_statistics = calculate_relevant_docs_statistics(relevant_docs)
        top_ranked_statistics = (
            calculate_top_ranked_statistics(top_ranked, num_queries)
            if top_ranked is not None and num_queries and len(top_ranked) > 0
            else None
        )

        return RetrievalDescriptiveStatistics(
            num_samples=num_documents + num_queries,
            num_queries=num_queries,
            num_documents=num_documents,
            number_of_characters=number_of_characters,
            documents_text_statistics=corpus_stats["text_statistics"],
            documents_image_statistics=corpus_stats["image_statistics"],
            documents_audio_statistics=corpus_stats["audio_statistics"],
            documents_video_statistics=corpus_stats["video_statistics"],
            queries_text_statistics=queries_stats["text_statistics"],
            queries_image_statistics=queries_stats["image_statistics"],
            queries_audio_statistics=queries_stats["audio_statistics"],
            queries_video_statistics=queries_stats["video_statistics"],
            relevant_docs_statistics=relevant_docs_statistics,
            top_ranked_statistics=top_ranked_statistics,
        )

    def _push_dataset_to_hub(
        self,
        repo_name: str,
        num_proc: int | None = None,
        **kwargs: Any,
    ) -> None:
        self.convert_v1_dataset_format_to_v2(num_proc)

        def _push_section(
            data: dict[str, RetrievalSplitData],
            subset_item: Literal["corpus", "queries", "relevant_docs", "top_ranked"],
            hf_subset_name: str,
            converter: Callable[[Any, Any], dict[str, Any]] | None = None,
        ) -> None:
            """Helper function to push dataset

            Args:
                data: Dataset with all items
                subset_item: Select which part to take. E. g. corpus, queries etc
                hf_subset_name: Name of the current item on HF
                converter: Function to convert dict to datasets format
            """
            sections = {}
            for split, split_data in data.items():
                # skip empty instructions and top ranked
                if subset_item not in split_data or split_data[subset_item] is None:
                    continue
                if isinstance(split_data[subset_item], Dataset):
                    sections[split] = split_data[subset_item]
                elif converter is not None:
                    subset_data = split_data[subset_item]
                    if subset_data is None:
                        continue

                    sections[split] = Dataset.from_list(
                        [converter(idx, item) for idx, item in subset_data.items()]
                    )
                else:
                    raise ValueError(
                        f"Unexpected subset item type {subset_item} without converter"
                    )
            if len(sections) > 0:
                DatasetDict(sections).push_to_hub(
                    repo_name,
                    hf_subset_name,
                    commit_message=f"Add {hf_subset_name}-{subset_item}",
                    num_proc=num_proc,
                    **kwargs,
                )

        for subset in self.dataset:
            logger.info(f"Converting {subset} of {self.metadata.name}")
            _push_section(
                self.dataset[subset],
                "queries",
                f"{subset}-queries" if subset != "default" else "queries",
            )
            _push_section(
                self.dataset[subset],
                "corpus",
                f"{subset}-corpus" if subset != "default" else "corpus",
            )
            # Handle relevant_docs separately since one entry expands to multiple records.
            relevant_sections = {}
            for split, values in self.dataset[subset].items():
                relevant_docs = values["relevant_docs"]
                entries = []
                for query_id, docs in relevant_docs.items():
                    for doc_id, score in docs.items():
                        entries.append(
                            {
                                "query-id": query_id,
                                "corpus-id": doc_id,
                                "score": score,
                            }
                        )
                relevant_sections[split] = Dataset.from_list(entries)
            DatasetDict(relevant_sections).push_to_hub(
                repo_name,
                f"{subset}-qrels" if subset != "default" else "qrels",
                commit_message=f"Add {subset}-qrels",
                num_proc=num_proc,
            )

            _push_section(
                self.dataset[subset],
                "top_ranked",
                f"{subset}-top_ranked" if subset != "default" else "top_ranked",
                lambda idx, docs: {"query-id": idx, "corpus-ids": docs},
            )

    def convert_to_reranking(
        self,
        top_ranked_path: str | Path,
        top_k: int = 10,
    ) -> Self:
        """Converts a reranking task to re-ranking by loading predictions from previous model run where the `prediction_folder` was specified.

        Args:
            top_ranked_path: Path to file or folder with the top ranked predictions.
            top_k: Number of results to load.

        Returns:
            The current task reformulated as a reranking task

        Raises:
            FileNotFoundError: If the specified path does not exist.
            ValueError: If the loaded top ranked results are not in the expected format.
        """
        self._top_k = top_k

        top_ranked_path = Path(top_ranked_path)
        if top_ranked_path.is_dir():
            top_ranked_path = self._predictions_path(top_ranked_path)

        if not top_ranked_path.exists():
            raise FileNotFoundError(
                f"Can't find previous results for this task. File {top_ranked_path} does not exist."
            )

        with top_ranked_path.open("r") as previous_results_file:
            previous_results = json.load(previous_results_file)

        if not self.data_loaded:
            self.load_data()

        self._previous_results_model_meta = previous_results["mteb_model_meta"]

        for subset in self.dataset:
            for split in self.dataset[subset]:
                top_ranked: RetrievalOutputType = previous_results[subset][split]
                if not isinstance(top_ranked, dict):
                    raise ValueError("Previous top ranked results is not a dictionary.")

                top_k_sorted = defaultdict(list)
                for query_id, values in top_ranked.items():
                    sorted_keys = sorted(values, key=lambda k: values[k], reverse=True)
                    top_k_sorted[query_id] = sorted_keys[: self._top_k]

                self.dataset[subset][split]["top_ranked"] = top_k_sorted
        return self


def _process_relevant_docs(
    collection: Mapping[str, Mapping[str, int]],
    hf_subset: str,
    split: str,
) -> dict[str, dict[str, int]]:
    """Collections can contain overlapping ids in different splits. Prepend split and subset to avoid this

    Returns:
        A new collection with split and subset prepended to ids
    """
    return_collection = {}
    for query_id, relevant in collection.items():
        return_collection[f"{split}_{hf_subset}_{query_id}"] = {
            f"{split}_{hf_subset}_{doc_id}": value for doc_id, value in relevant.items()
        }
    return return_collection
