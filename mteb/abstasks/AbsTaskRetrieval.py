from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from time import time
from typing import Any, Callable, TypedDict

from datasets import Dataset, DatasetDict

from mteb.encoder_interface import Encoder
from mteb.types import HFSubset, ScoresDict
from mteb.types.statistics import DescriptiveStatistics, TextStatistics

from ..create_dataloaders import corpus_to_dict
from ..evaluation.evaluators import RetrievalEvaluator
from ..evaluation.evaluators.retrieval_metrics import make_score_dict
from .AbsTask import AbsTask
from .dataset_loaders import RetrievalDatasetLoader, RetrievalSplitData
from .statistics_calculation import calculate_text_statistics

logger = logging.getLogger(__name__)


class TopRankedStatistics(TypedDict):
    """Statistics for top ranked documents in a retrieval task.

    Attributes:
        num_top_ranked: Total number of top ranked documents across all queries.
        min_top_ranked_per_query: Minimum number of top ranked documents for any query.
        average_top_ranked_per_query: Average number of top ranked documents per query.
        max_top_ranked_per_query: Maximum number of top ranked documents for any query.
    """

    num_top_ranked: int
    min_top_ranked_per_query: int
    average_top_ranked_per_query: float
    max_top_ranked_per_query: int


class RelevantDocsStatistics(TypedDict):
    """Statistics for relevant documents in a retrieval task.

    Attributes:
        num_relevant_docs: Total number of relevant documents across all queries.
        min_relevant_docs_per_query: Minimum number of relevant documents for any query.
        average_relevant_docs_per_query: Average number of relevant documents per query.
        max_relevant_docs_per_query: Maximum number of relevant documents for any query.
        unique_relevant_docs: Number of unique relevant documents across all queries.
    """

    num_relevant_docs: int
    min_relevant_docs_per_query: int
    average_relevant_docs_per_query: float
    max_relevant_docs_per_query: float
    unique_relevant_docs: int


class RetrievalDescriptiveStatistics(DescriptiveStatistics):
    """Descriptive statistics for Retrieval

    Attributes:
        num_samples: Number of queries and documents
        number_of_characters: Total number of characters in queries and documents

        documents_statistics: Statistics for documents
        queries_statistics: Statistics for queries
        relevant_docs_statistics: Statistics for relevant documents
        instructions_statistics: Statistics for instructions (if available)
        top_ranked_statistics: Statistics for top ranked documents (if available)
    """

    num_samples: int
    number_of_characters: int

    documents_statistics: TextStatistics
    queries_statistics: TextStatistics

    relevant_docs_statistics: RelevantDocsStatistics

    # these are for datasets with instructions
    instructions_statistics: TextStatistics | None

    # this is for datasets that do reranking
    top_ranked_statistics: TopRankedStatistics | None


class AbsTaskRetrieval(AbsTask):
    """Abstract class for retrieval experiments.

    Child-classes must implement the following properties:

    self.dataset: dict[str, dict[str, dict[str, Any]]]
        A dictionary containing all dataset components with the following structure:
        {
            hf_subset: {
                split: {
                    'corpus': dict[str, dict[str, str]],  # doc_id -> doc
                        Semantically, it should contain dict[split_name, dict[sample_id, dict[str, str]]]
                        E.g. {"test": {"document_one": {"_id": "d1", "title": "title", "text": "text"}}}
                    'queries': dict[str, str | list[str]]],  # query_id -> query
                        Semantically, it should contain dict[split_name, dict[sample_id, str]] or dict[split_name, dict[sample_id, list[str]]] for conversations
                        E.g. {"test": {"q1": "query"}}
                        or {"test": {"q1": ["turn1", "turn2", "turn3"]}}
                    'relevant_docs': dict[str, dict[str, int]],  # query_id -> doc_id -> score
                        Semantically, it should contain dict[split_name, dict[sample_id, dict[doc_id, score]]]
                        E.g.: {"test": {"q1": {"document_one": 1}}}
                    'instructions': Optional[dict[str, str]],  # query_id -> instruction
                        Semantically, it should contain dict[split_name, dict[sample_id, list[doc_id]]] or dict[split_name, dict[sample_id, dict[doc_id, score]]]
                        E.g.: {"test": {"q1": ["document_one", "document_two"]}} or {"test": {"q1": {"document_one": 1, "document_two": 0.5}}}
                    'top_ranked': Optional[dict[str, list[str]]]  # query_id -> doc_ids
                        Semantically, it should contain dict[split_name, dict[sample_id, str]]. If there are multiple instructions per query, please duplicate the queries and give them unique ids for consolidation.
                        E.g. {"test": {"query-id1": "instruction text"}}
                }
            }
        }
    """

    ignore_identical_ids: bool = False
    abstask_prompt = "Retrieve text based on user query."
    k_values: list[int] = [1, 3, 5, 10, 20, 100, 1000]
    top_k: int = max(k_values)
    dataset: dict[str, dict[str, RetrievalSplitData]]
    cross_encoder_top_k: int = 100

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dataset = defaultdict(
            lambda: defaultdict(
                lambda: RetrievalSplitData(
                    corpus={},
                    queries={},
                    relevant_docs={},
                    instructions=None,
                    top_ranked=None,
                )
            )
        )

    def convert_v1_dataset_format_to_v2(self):
        if not hasattr(self, "queries"):
            return
        self.dataset = defaultdict(
            lambda: defaultdict(
                lambda: RetrievalSplitData(
                    corpus={},
                    queries={},
                    relevant_docs={},
                    instructions=None,
                    top_ranked=None,
                )
            )
        )

        if self.metadata.is_multilingual:
            for subset in self.queries:
                for split in self.queries[subset]:
                    self.dataset[subset][split]["queries"] = self.queries[subset][split]
                    self.dataset[subset][split]["corpus"] = self.corpus[subset][split]
                    self.dataset[subset][split]["relevant_docs"] = self.relevant_docs[
                        subset
                    ][split]
                    if hasattr(self, "instructions"):
                        self.dataset[subset][split]["instructions"] = self.instructions[
                            subset
                        ][split]
                    if hasattr(self, "top_ranked"):
                        self.dataset[subset][split]["top_ranked"] = self.top_ranked[
                            subset
                        ][split]
        else:
            subset = "default"
            for split in self.queries:
                self.dataset[subset][split]["queries"] = self.queries[split].copy()
                self.dataset[subset][split]["corpus"] = self.corpus[split].copy()
                self.dataset[subset][split]["relevant_docs"] = self.relevant_docs[
                    split
                ].copy()
                if hasattr(self, "instructions"):
                    self.dataset[subset][split]["instructions"] = self.instructions[
                        split
                    ].copy()
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

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        dataset_path = self.metadata.dataset["path"]
        eval_splits = kwargs.get("eval_splits", self.metadata.eval_splits)
        trust_remote_code = self.metadata.dataset.get("trust_remote_code", False)
        revision = self.metadata.dataset["revision"]

        def process_data(split: str, hf_subset: str = "default"):
            """Helper function to load and process data for a given split and language"""
            logger.info(
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
                    process_data(split, lang)
        else:
            for split in eval_splits:
                process_data(split)
        self.dataset_transform()
        self.data_loaded = True

    def evaluate(
        self,
        model: Encoder,
        split: str = "test",
        subsets_to_run: list[HFSubset] | None = None,
        *,
        encode_kwargs: dict[str, Any],
        **kwargs,
    ) -> dict[HFSubset, ScoresDict]:
        """Evaluate the model on the retrieval task.

        Args:
            model: Model to evaluate
            split: Split to evaluate on
            subsets_to_run: Optional list of subsets to evaluate on
            encode_kwargs: Keyword arguments passed to the encoder
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
            **kwargs,
        )

    def _evaluate_subset(
        self,
        model: Encoder,
        data_split: DatasetDict | Dataset,
        encode_kwargs: dict[str, Any],
        hf_split: str,
        hf_subset: str,
        # retrieval specific args
        save_predictions: bool = False,
        export_errors: bool = False,
        save_qrels: bool = False,
        output_folder: str = "results",
        results: dict[str, dict[str, float]] | None = None,
        **kwargs,
    ) -> ScoresDict:
        """Evaluate a model on a specific subset of the data.

        Args:
            model: Model to evaluate
            data_split: Data split to evaluate on
            encode_kwargs: Keyword arguments passed to the encoder
            hf_split: Split to evaluate on
            hf_subset: Subset to evaluate on
            save_predictions: Whether to save predictions
            export_errors: Whether to export errors
            save_qrels: Whether to save the qrels
            output_folder: Folder to save the results
            results: Results from retrieval from previous run
            **kwargs: Additional keyword arguments passed to the evaluator

        Returns:
            Dictionary of evaluation scores
        """
        retriever = RetrievalEvaluator(
            corpus=data_split["corpus"],
            queries=data_split["queries"],
            task_metadata=self.metadata,
            hf_split=hf_split,
            hf_subset=hf_subset,
            instructions=data_split["instructions"],
            top_ranked=data_split["top_ranked"],
            top_k=self.top_k,
            **kwargs,
        )

        if not results:
            start_time = time()
            results = retriever(
                model,
                encode_kwargs=encode_kwargs,
                **kwargs,
            )
            end_time = time()
            logger.debug(f"Time taken to retrieve: {end_time - start_time:.2f} seconds")

        if save_predictions or export_errors or save_qrels:
            output_folder = Path(output_folder)
            if not output_folder.exists():
                output_folder.mkdir(parents=True)

        if save_predictions:
            if self.top_k is not None:
                for qid in list(results.keys()):
                    doc_ids = set(
                        sorted(
                            results[qid], key=lambda x: results[qid][x], reverse=True
                        )[: self.top_k]
                    )
                    results[qid] = {
                        k: v for k, v in results[qid].items() if k in doc_ids
                    }
            qrels_save_path = (
                output_folder / f"{self.metadata.name}_{hf_subset}_predictions.json"
            )

            with open(qrels_save_path, "w") as f:
                json.dump(results, f)

        if save_qrels:
            with open(
                output_folder / f"{self.metadata.name}_{hf_subset}_qrels.json", "w"
            ) as f:
                json.dump(data_split["relevant_docs"], f)

        (
            all_scores,
            ndcg,
            _map,
            recall,
            precision,
            naucs,
            mrr,
            naucs_mrr,
        ) = retriever.evaluate(
            data_split["relevant_docs"],
            results,
            self.k_values,
            ignore_identical_ids=self.ignore_identical_ids,
        )
        task_specific_scores = self.task_specific_scores(
            all_scores,
            data_split["relevant_docs"],
            results,
            hf_split=hf_split,
            hf_subset=hf_subset,
        )
        scores = make_score_dict(
            ndcg, _map, recall, precision, mrr, naucs, naucs_mrr, task_specific_scores
        )

        if export_errors:
            errors = {}

            if not save_predictions:
                for qid in results.keys():
                    doc_scores = results[qid]
                    sorted_docs = sorted(
                        doc_scores.items(), key=lambda x: x[1], reverse=True
                    )[:1]
                    results[qid] = dict(sorted_docs)
            for qid, retrieved_docs in results.items():
                expected_docs = data_split["relevant_docs"][qid]
                false_positives = [
                    doc for doc in retrieved_docs if doc not in expected_docs
                ]
                false_negatives = [
                    doc for doc in expected_docs if doc not in retrieved_docs
                ]
                if false_positives or false_negatives:
                    errors[qid] = {
                        "false_positives": false_positives,
                        "false_negatives": false_negatives,
                    }
            errors_save_path = (
                output_folder / f"{self.metadata.name}_{hf_subset}_errors.json"
            )
            with errors_save_path.open("w") as f:
                json.dump(errors, f)

        return scores

    def task_specific_scores(
        self,
        scores: dict[str, dict[str, float]],
        qrels: dict[str, dict[str, int]],
        results: dict[str, dict[str, float]],
        hf_split: str,
        hf_subset: str,
    ) -> dict[str, float]:
        return {}

    def _calculate_metrics_from_split(
        self, split: str, hf_subset: str | None = None, compute_overall: bool = False
    ) -> RetrievalDescriptiveStatistics:
        self.convert_v1_dataset_format_to_v2()
        if hf_subset and hf_subset in self.dataset:
            split_data = self.dataset[hf_subset][split]
            queries = split_data["queries"]
            corpus = split_data["corpus"]
            relevant_docs = split_data["relevant_docs"]
            instructions = split_data["instructions"]
            top_ranked = split_data["top_ranked"]
        elif compute_overall:
            queries = {}
            corpus = {}
            relevant_docs = {}
            instructions = {}
            top_ranked = {}
            for hf_subset in self.metadata.eval_langs:
                split_data = self.dataset[hf_subset][split]
                queries.update(process_docs(split_data["queries"], hf_subset, split))
                corpus.update(process_docs(split_data["corpus"], hf_subset, split))
                relevant_docs.update(
                    process_relevant_docs(split_data["relevant_docs"], hf_subset, split)
                )
                if (
                    "instructions" in split_data
                    and split_data["instructions"] is not None
                ):
                    instructions.update(
                        process_docs(split_data["instructions"], hf_subset, split)
                    )
                if "top_ranked" in split_data and split_data["top_ranked"] is not None:
                    top_ranked.update(
                        process_docs(split_data["top_ranked"], hf_subset, split)
                    )
        else:
            if "default" in self.dataset and split != "default":
                return self._calculate_metrics_from_split(
                    split=split, hf_subset="default"
                )
            split_data = self.dataset["default"][split]
            queries = split_data["queries"]
            corpus = split_data["corpus"]
            relevant_docs = split_data["relevant_docs"]
            instructions = split_data["instructions"]
            top_ranked = split_data["top_ranked"]

        corpus = list(corpus_to_dict(list(corpus.values()))["text"])
        # todo handle conversations e. g. Statcan
        queries_texts = [q for q in queries.values() if isinstance(q, str)]
        num_documents = len(corpus)
        num_queries = len(queries_texts)

        # create a list of number of relevant docs per query
        qrels_lengths = [
            len(relevant_docs[qid]) for qid in relevant_docs if qid in queries
        ]
        unique_qrels = len({doc for qid in relevant_docs for doc in relevant_docs[qid]})
        # number of qrels that are not 0
        num_qrels_non_zero = sum(
            sum(1 for doc_id in docs if docs[doc_id] != 0)
            for docs in relevant_docs.values()
        )
        qrels_per_doc = num_qrels_non_zero / len(relevant_docs) if num_queries else 0

        relevant_docs_statistics = RelevantDocsStatistics(
            num_relevant_docs=num_qrels_non_zero,
            min_relevant_docs_per_query=min(qrels_lengths),
            average_relevant_docs_per_query=qrels_per_doc,
            max_relevant_docs_per_query=max(qrels_lengths),
            unique_relevant_docs=unique_qrels,
        )

        if instructions is not None and len(instructions) > 0:
            instruction_statistics = calculate_text_statistics(instructions)
        else:
            instruction_statistics = None

        if top_ranked is not None and num_queries and len(top_ranked) > 0:
            top_ranked_statistics = TopRankedStatistics(
                num_top_ranked=sum(
                    len(docs) for docs in top_ranked.values() if docs is not None
                ),
                min_top_ranked_per_query=min(
                    len(docs) for docs in top_ranked.values() if docs is not None
                ),
                average_top_ranked_per_query=(
                    sum(len(docs) for docs in top_ranked.values() if docs is not None)
                    / num_queries
                ),
                max_top_ranked_per_query=max(
                    len(docs) for docs in top_ranked.values() if docs is not None
                ),
            )
        else:
            top_ranked_statistics = None

        corpus_statistics = calculate_text_statistics(corpus)
        queries_statistics = calculate_text_statistics(queries)

        number_of_characters = (
            corpus_statistics["total_text_length"]
            + queries_statistics["total_text_length"]
        )
        if instruction_statistics is not None:
            number_of_characters += instruction_statistics["total_text_length"]

        return RetrievalDescriptiveStatistics(
            num_samples=num_documents + num_queries,
            number_of_characters=number_of_characters,
            documents_statistics=corpus_statistics,
            queries_statistics=queries_statistics,
            relevant_docs_statistics=relevant_docs_statistics,
            instructions_statistics=instruction_statistics,
            top_ranked_statistics=top_ranked_statistics,
        )

    def _push_dataset_to_hub(self, repo_name: str) -> None:
        def format_text_field(text: str | dict[str, str]) -> str:
            if isinstance(text, str):
                return text
            return (
                f"{text['title']} {text['text']}".strip()
                if text.get("title", None) is not None
                else text["text"]
            )

        def push_section(
            data: dict[str, dict[Any, Any]],
            column: str,
            suffix: str,
            converter: Callable[[Any, Any], dict[str, Any]],
        ) -> None:
            sections = {}
            for split in data.keys():
                # skip empty instructions and top ranked
                if column not in data[split] or data[split][column] is None:
                    continue
                sections[split] = Dataset.from_list(
                    [converter(idx, item) for idx, item in data[split][column].items()]
                )
            if len(sections) > 0:
                DatasetDict(sections).push_to_hub(repo_name, suffix)

        for subset in self.dataset:
            logger.info(f"Converting {subset} of {self.metadata.name}")
            push_section(
                self.dataset[subset],
                "queries",
                f"{subset}-queries" if subset != "default" else "queries",
                lambda idx, text: {"_id": idx, "text": text},
            )
            push_section(
                self.dataset[subset],
                "corpus",
                f"{subset}-corpus" if subset != "default" else "corpus",
                lambda idx, text: {
                    "_id": idx,
                    "text": format_text_field(text),
                    "title": text.get("title", "") if isinstance(text, dict) else "",
                },
            )
            # Handle relevant_docs separately since one entry expands to multiple records.
            relevant_sections = {}
            for split, values in self.dataset[subset].items():
                relecant_docs = values["relevant_docs"]
                entries = []
                for query_id, docs in relecant_docs.items():
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
                repo_name, f"{subset}-qrels" if subset != "default" else "qrels"
            )

            push_section(
                self.dataset[subset],
                "instructions",
                f"{subset}-instruction" if subset != "default" else "instruction",
                lambda idx, text: {"query-id": idx, "instruction": text},
            )
            push_section(
                self.dataset[subset],
                "top_ranked",
                f"{subset}-top_ranked" if subset != "default" else "top_ranked",
                lambda idx, docs: {"query-id": idx, "corpus-ids": docs},
            )


def calculate_queries_length(queries: dict[str, str]) -> list[int] | None:
    queries_lens = []
    for query in queries.values():
        if query is None or len(query) == 0:
            continue

        if isinstance(query[0], str):
            queries_lens.append(len(query))
        else:
            queries_lens.extend([len(turn) for turn in query])
    return queries_lens


def calculate_corpus_length(
    corpus: dict[str, str | dict[str, str]],
) -> list[int] | None:
    doc_lens = []
    if corpus is None:
        return None
    for doc in corpus.values():
        if isinstance(doc, dict):
            doc_lens.append(len(doc["text"]) + len(doc.get("title", "")))
        else:
            doc_lens.append(len(doc))

    return doc_lens


def process_docs(
    collection: dict[str, dict[str, str] | str], hf_subset: str, split: str
) -> dict[str, str]:
    """Collections can contain overlapping ids in different splits. Prepend split to avoid this"""
    return {f"{split}_{hf_subset}_{k}": v for k, v in collection.items()}


def process_relevant_docs(
    collection: dict[str, dict[str, int]],
    hf_subset: str,
    split: str,
) -> dict[str, dict[str, int]]:
    """Collections can contain overlapping ids in different splits. Prepend split to avoid this"""
    return_collection = {}
    for query_id, relevant in collection.items():
        return_collection[f"{split}_{hf_subset}_{query_id}"] = {
            f"{split}_{hf_subset}_{doc_id}": value for doc_id, value in relevant.items()
        }
    return return_collection
