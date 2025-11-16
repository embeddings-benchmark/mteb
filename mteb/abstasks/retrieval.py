import json
import logging
from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from time import time
from typing import Any, Literal

from datasets import Dataset, DatasetDict, concatenate_datasets
from typing_extensions import Self

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
    MTEBModels,
    SearchCrossEncoderWrapper,
    SearchEncoderWrapper,
    SearchProtocol,
)
from mteb.types import (
    HFSubset,
    QueryDatasetType,
    RelevantDocumentsType,
    RetrievalOutputType,
    ScoresDict,
)
from mteb.types.statistics import (
    ImageStatistics,
    RelevantDocsStatistics,
    SplitDescriptiveStatistics,
    TextStatistics,
    TopRankedStatistics,
)

from ._statistics_calculation import (
    calculate_image_statistics,
    calculate_relevant_docs_statistics,
    calculate_text_statistics,
    calculate_top_ranked_statistics,
)
from .abstask import AbsTask
from .retrieval_dataset_loaders import (
    RetrievalDatasetLoader,
    RetrievalSplitData,
    _combine_queries_with_instructions_datasets,
)

logger = logging.getLogger(__name__)


class RetrievalDescriptiveStatistics(SplitDescriptiveStatistics):
    """Descriptive statistics for Retrieval

    Attributes:
        num_samples: Number of queries and documents
        number_of_characters: Total number of characters in queries and documents

        documents_text_statistics: Statistics for documents
        documents_image_statistics: Statistics for documents
        queries_text_statistics: Statistics for queries
        queries_image_statistics: Statistics for queries
        relevant_docs_statistics: Statistics for relevant documents
        top_ranked_statistics: Statistics for top ranked documents (if available)
    """

    num_samples: int
    number_of_characters: int

    documents_text_statistics: TextStatistics | None
    documents_image_statistics: ImageStatistics | None
    queries_text_statistics: TextStatistics | None
    queries_image_statistics: ImageStatistics | None

    relevant_docs_statistics: RelevantDocsStatistics

    # this is for datasets that do reranking
    top_ranked_statistics: TopRankedStatistics | None


def _filter_queries_without_positives(
    relevant_docs: RelevantDocumentsType, queries: QueryDatasetType
) -> tuple[RelevantDocumentsType, QueryDatasetType]:
    _relevant_docs = {}
    for idx in relevant_docs:
        if len(relevant_docs[idx]) == 0:  # no relevant docs
            continue
        _relevant_docs[idx] = relevant_docs[idx]

    queries = queries.filter(
        lambda x: x["id"] in _relevant_docs.keys(), desc="Filtering queries by qrels"
    )

    return _relevant_docs, queries


class AbsTaskRetrieval(AbsTask):
    """Abstract class for retrieval experiments.

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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        empty_dataset = Dataset.from_dict({})
        self.dataset = defaultdict(
            lambda: defaultdict(
                lambda: RetrievalSplitData(
                    corpus=empty_dataset,
                    queries=empty_dataset,
                    relevant_docs={},
                    top_ranked=None,
                )
            )
        )

    def convert_v1_dataset_format_to_v2(self):
        """Convert dataset from v1 (from `self.queries`, `self.document`) format to v2 format (`self.dotaset`)."""
        # check if dataset is `v1` version
        if not hasattr(self, "queries"):
            return
        empty_dataset = Dataset.from_dict({})

        self.dataset = defaultdict(
            lambda: defaultdict(
                lambda: RetrievalSplitData(
                    corpus=empty_dataset,
                    queries=empty_dataset,
                    relevant_docs={},
                    top_ranked=None,
                )
            )
        )

        def _process_split(
            ds_queries: dict | Dataset, ds_corpus: dict | Dataset
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
                for split in self.queries[subset]:
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
                            )
                        )
                    if hasattr(self, "top_ranked"):
                        self.dataset[subset][split]["top_ranked"] = self.top_ranked[
                            subset
                        ][split]
        else:
            subset = "default"
            for split in self.queries:
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
                        )
                    )
                if hasattr(self, "top_ranked"):
                    self.dataset[subset][split]["top_ranked"] = self.top_ranked[
                        split
                    ].copy()

        del self.queries
        del self.corpus
        del self.relevant_docs
        if hasattr(self, "instructions"):
            del self.instructions
        if hasattr(self, "top_ranked"):
            del self.top_ranked

    def load_data(self) -> None:
        """Load the dataset for the retrieval task."""
        if self.data_loaded:
            return

        dataset_path = self.metadata.dataset["path"]
        eval_splits = self.metadata.eval_splits
        trust_remote_code = self.metadata.dataset.get("trust_remote_code", False)
        revision = self.metadata.dataset["revision"]

        def _process_data(split: str, hf_subset: str = "default"):
            """Helper function to load and process data for a given split and language"""
            logger.debug(
                f"Loading {split} split for {hf_subset} subset of {self.metadata.name}"
            )

            self.dataset[hf_subset][split] = RetrievalDatasetLoader(
                hf_repo=dataset_path,
                revision=revision,
                trust_remote_code=trust_remote_code,
                split=split,
                config=hf_subset,
            ).load()

        if self.metadata.is_multilingual:
            for lang in self.metadata.eval_langs:
                for split in eval_splits:
                    _process_data(split, lang)
        else:
            for split in eval_splits:
                _process_data(split)
        self.dataset_transform()
        self.data_loaded = True

    def evaluate(
        self,
        model: MTEBModels,
        split: str = "test",
        subsets_to_run: list[HFSubset] | None = None,
        *,
        encode_kwargs: dict[str, Any],
        prediction_folder: Path | None = None,
        **kwargs,
    ) -> dict[HFSubset, ScoresDict]:
        """Evaluate the model on the retrieval task.

        Args:
            model: Model to evaluate. Model should implement the [SearchProtocol][mteb.models.models_protocols.SearchProtocol]
                or be an [Encoder][mteb.models.models_protocols.EncoderProtocol] or [CrossEncoderProtocol][mteb.models.models_protocols.CrossEncoderProtocol].
            split: Split to evaluate on
            subsets_to_run: Optional list of subsets to evaluate on
            encode_kwargs: Keyword arguments passed to the encoder
            prediction_folder: Folder to save model predictions
            **kwargs: Additional keyword arguments passed to the evaluator


        Returns:
            Dictionary mapping subsets to their evaluation scores
        """
        if not self.data_loaded:
            self.load_data()
        # TODO: convert all tasks directly https://github.com/embeddings-benchmark/mteb/issues/2030
        self.convert_v1_dataset_format_to_v2()

        return super().evaluate(
            model,
            split,
            subsets_to_run,
            encode_kwargs=encode_kwargs,
            prediction_folder=prediction_folder,
            **kwargs,
        )

    def _evaluate_subset(
        self,
        model: MTEBModels,
        data_split: RetrievalSplitData,
        encode_kwargs: dict[str, Any],
        hf_split: str,
        hf_subset: str,
        prediction_folder: Path | None = None,
        **kwargs,
    ) -> ScoresDict:
        """Evaluate a model on a specific subset of the data.

        Args:
            model: Model to evaluate
            data_split: Data split to evaluate on
            encode_kwargs: Keyword arguments passed to the encoder
            hf_split: Split to evaluate on
            hf_subset: Subset to evaluate on
            prediction_folder: Folder with results prediction
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
            **kwargs,
        )

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

        start_time = time()
        results = retriever(
            search_model,
            encode_kwargs=encode_kwargs,
        )
        end_time = time()
        logger.debug(
            f"Running retrieval task - Time taken to retrieve: {end_time - start_time:.2f} seconds"
        )

        if prediction_folder:
            self._save_task_predictions(
                results,
                model,
                prediction_folder,
                hf_subset=hf_subset,
                hf_split=hf_split,
            )

        logger.info("Running retrieval task - Evaluating retrieval scores...")
        (
            all_scores,
            ndcg,
            _map,
            recall,
            precision,
            naucs,
            mrr,
            naucs_mrr,
            cv_recall,
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
            ndcg,
            _map,
            recall,
            precision,
            mrr,
            naucs,
            naucs_mrr,
            cv_recall,
            task_specific_scores,
            self._previous_results_model_meta,
        )

    def task_specific_scores(
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

    def _calculate_descriptive_statistics_from_split(
        self, split: str, hf_subset: str | None = None, compute_overall: bool = False
    ) -> RetrievalDescriptiveStatistics:
        self.convert_v1_dataset_format_to_v2()
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
            for hf_subset in self.metadata.eval_langs:
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
            queries_modalities = "t"
            corpus_modalities = "t"
        else:
            queries_modalities, corpus_modalities = self.metadata.category.split("2")

        number_of_characters = 0

        documents_text_statistics = None
        documents_image_statistics = None
        queries_text_statistics = None
        queries_image_statistics = None

        if "t" in corpus_modalities:
            corpus_texts = corpus.map(_corpus_to_dict)["text"]
            documents_text_statistics = calculate_text_statistics(corpus_texts)
            number_of_characters += documents_text_statistics["total_text_length"]

        if "i" in corpus_modalities:
            documents_image_statistics = calculate_image_statistics(corpus["image"])

        if "t" in queries_modalities:
            queries_ = queries
            if "instruction" in queries_[0]:
                queries_ = queries_.map(_combine_queries_with_instruction_text)

            if isinstance(queries_["text"][0], dict | list):
                queries_ = queries_.map(_convert_conv_history_to_query)
            queries_text_statistics = calculate_text_statistics(queries_["text"])

            number_of_characters += queries_text_statistics["total_text_length"]

        if "i" in queries_modalities:
            queries_image_statistics = calculate_image_statistics(queries["image"])

        relevant_docs_statistics = calculate_relevant_docs_statistics(relevant_docs)

        if top_ranked is not None and num_queries and len(top_ranked) > 0:
            top_ranked_statistics = calculate_top_ranked_statistics(
                top_ranked, num_queries
            )
        else:
            top_ranked_statistics = None

        return RetrievalDescriptiveStatistics(
            num_samples=num_documents + num_queries,
            number_of_characters=number_of_characters,
            documents_text_statistics=documents_text_statistics,
            documents_image_statistics=documents_image_statistics,
            queries_text_statistics=queries_text_statistics,
            queries_image_statistics=queries_image_statistics,
            relevant_docs_statistics=relevant_docs_statistics,
            top_ranked_statistics=top_ranked_statistics,
        )

    def _push_dataset_to_hub(self, repo_name: str) -> None:
        self.convert_v1_dataset_format_to_v2()

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
            for split in data.keys():
                # skip empty instructions and top ranked
                if subset_item not in data[split] or data[split][subset_item] is None:
                    continue
                if isinstance(data[split][subset_item], Dataset):
                    sections[split] = data[split][subset_item]
                elif converter is not None:
                    sections[split] = Dataset.from_list(
                        [
                            converter(idx, item)
                            for idx, item in data[split][subset_item].items()
                        ]
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
                    sorted_keys = sorted(values, key=values.get, reverse=True)
                    top_k_sorted[query_id] = sorted_keys[: self._top_k]

                self.dataset[subset][split]["top_ranked"] = top_k_sorted
        return self


def _process_relevant_docs(
    collection: dict[str, dict[str, float]],
    hf_subset: str,
    split: str,
) -> dict[str, dict[str, float]]:
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
