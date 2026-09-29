from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from datasets import (
    Dataset,
    DatasetDict,
    concatenate_datasets,
    get_dataset_config_names,
    get_dataset_split_names,
    load_dataset,
)

from mteb._create_dataloaders import (
    _combine_queries_with_instruction_text,
    _convert_conv_history_to_query,
    _corpus_to_dict,
)
from mteb._evaluators import RetrievalEvaluator
from mteb._evaluators.retrieval_metrics import make_score_dict, ndcg_float_scores
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

from ._statistics_calculation import (
    calculate_relevant_docs_statistics,
    calculate_single_input_modality_statistics,
    calculate_top_ranked_statistics,
)
from .abstask import AbsTask
from .retrieval_dataset_loaders import (
    RetrievalDatasetLoader,
    _combine_queries_with_instructions_datasets,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from typing_extensions import Self

    from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData
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
    dataset: dict[str, dict[str, RetrievalSplitData]]
    _support_cross_encoder: bool = True
    _support_search: bool = True
    _previous_results_model_meta: dict[str, Any] | None = None
    skip_first_result: bool = False

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._top_k: int = max(self.k_values)

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

    def _get_content_columns(self) -> dict[str, Modalities]:
        """The corpus and query columns holding the documents, mapped to their modality.

        Retrieval stores each modality in a column named after the modality itself. Text also carries an optional
        `title`, which is part of the document: a corpus entry is encoded as `"{title} {text}"`. Queries have no
        title, so a filter compares whichever of these columns the corpus and the queries actually have.
        """
        columns: dict[str, Modalities] = {}
        for modality in self.metadata.modalities:
            if modality == "text":
                columns["title"] = "text"
                columns["text"] = "text"
            else:
                columns[modality] = modality
        return columns

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
            query_ids = set(queries["id"])
            corpus_ids = set(corpus["id"])
        elif compute_overall:
            queries = None
            corpus = None
            relevant_docs = {}
            top_ranked = {}
            query_ids = set()
            corpus_ids = set()
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

                query_ids.update(
                    f"{split}_{hf_subset}_{query_id}"
                    for query_id in split_data["queries"]["id"]
                )
                corpus_ids.update(
                    f"{split}_{hf_subset}_{corpus_id}"
                    for corpus_id in split_data["corpus"]["id"]
                )
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
            query_ids = set(queries["id"])
            corpus_ids = set(corpus["id"])

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

        relevant_docs_statistics = calculate_relevant_docs_statistics(
            relevant_docs, query_ids, corpus_ids
        )
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
      ``{subset}-excluded`` config (``query-id``, ``excluded-corpus-ids``). Both metric families are
      reported on the full-corpus ranking: the integer-qrels metrics, and ``ndcg_float_at_k`` over
      the gains (the gains apply to the judged pool documents; unjudged documents score 0).
      Cross-encoders are refused in this mode (they would score every
      query against the whole corpus).

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
        """Restrict the corpus to the reranking pool; gains are loaded in ``task_specific_scores``."""
        for hf_subset, splits in self.dataset.items():
            for split, data in splits.items():
                if not self.rerank_top_ranked:
                    data["top_ranked"] = self._full_corpus_candidates(
                        hf_subset, split, data, num_proc
                    )
                elif self.restrict_corpus_to_top_ranked:
                    top_ranked = data.get("top_ranked")
                    if top_ranked:
                        keep = {
                            doc_id for docs in top_ranked.values() for doc_id in docs
                        }
                        corpus = data["corpus"]
                        data["corpus"] = corpus.select(
                            [
                                i
                                for i, doc_id in enumerate(corpus["id"])
                                if doc_id in keep
                            ]
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

        It gets its own name (``<name>.retrieval``) so its results are not stored under, and do
        not overwrite, the reranking task's results. The main score is unchanged: both the
        integer-qrels metrics and ``ndcg_float_at_k`` are reported on the full-corpus ranking
        (the gains apply to the judged pool documents; unjudged documents score 0).
        """
        task = type(self)()
        task.rerank_top_ranked = False
        task.restrict_corpus_to_top_ranked = False
        task.metadata = self.metadata.model_copy(
            update={
                "name": f"{self.metadata.name}.retrieval",
                "description": f"{self.metadata.description} Full-corpus retrieval view: the"
                " ranking covers the whole corpus; the gains apply to the judged pool documents"
                " and unjudged documents score 0, next to the integer-qrels metrics.",
            }
        )
        return task

    @property
    def _support_cross_encoder(self) -> bool:  # type: ignore[override]
        # a cross-encoder would score every (query, document) pair of the full corpus
        return self.rerank_top_ranked

    def task_specific_scores(
        self,
        scores: dict[str, dict[str, float]],
        qrels: RelevantDocumentsType,
        results: dict[str, dict[str, float]],
        hf_split: str,
        hf_subset: str,
    ) -> dict[str, float]:
        """Adds ``ndcg_float_at_k`` over the float gains, for the queries the qrels metrics score.

        Computed in both modes: over the pool-restricted ranking when reranking, and over the
        full-corpus ranking otherwise (the gains apply to the judged pool documents; unjudged
        documents score 0).
        """
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
