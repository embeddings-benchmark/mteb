from __future__ import annotations

import json
import logging
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from time import time
from typing import Any, Callable, TypedDict

from datasets import Dataset, DatasetDict

from mteb.abstasks.TaskMetadata import DescriptiveStatistics, HFSubset
from mteb.encoder_interface import Encoder

from ..evaluation.evaluators import RetrievalEvaluator
from ..evaluation.evaluators.utils import make_score_dict
from ..load_results.task_results import ScoresDict
from .AbsTask import AbsTask
from .dataloaders import RetrievalDataLoader

logger = logging.getLogger(__name__)


class RetrievalDescriptiveStatistics(DescriptiveStatistics):
    """Descriptive statistics for Retrieval

    Attributes:
        num_samples: Number of queries and documents
        num_relevant_docs: Number of relevant documents

        num_documents: Number of documents
        min_document_length: Minimum length of documents
        average_document_length: Average length of documents
        max_document_length: Maximum length of documents
        unique_documents: Number of unique documents

        num_queries: number of queries in the dataset
        min_query_length: Minimum length of queries
        average_query_length: Average length of queries
        max_query_length: Maximum length of queries
        unique_queries: Number of unique queries
        none_queries: Number of none queries

        number_of_characters: Total number of symbols in the dataset
        min_relevant_docs_per_query: Minimum number of relevant documents per query
        average_relevant_docs_per_query: Average number of relevant documents per query
        max_relevant_docs_per_query: Maximum number of relevant documents per query
        unique_relevant_docs: Number of unique relevant documents

        num_instructions: Number of instructions
        min_instruction_length: Minimum length of instructions
        average_instruction_length: Average length of instructions
        max_instruction_length: Maximum length of instructions
        unique_instructions: Number of unique instructions

        num_top_ranked: Number of top ranked documents
        min_top_ranked_per_query: Minimum number of top ranked documents per query
        average_top_ranked_per_query: Average number of top ranked documents per query
        max_top_ranked_per_query: Maximum number of relevant documents per query
    """

    num_samples: int
    number_of_characters: int

    num_documents: int
    min_document_length: int
    average_document_length: float
    max_document_length: int
    unique_documents: int

    num_queries: int
    min_query_length: int
    average_query_length: float
    max_query_length: int
    unique_queries: int
    none_queries: int

    num_relevant_docs: int
    min_relevant_docs_per_query: int
    average_relevant_docs_per_query: float
    max_relevant_docs_per_query: float
    unique_relevant_docs: int

    # these are for datasets with instructions
    num_instructions: int | None
    min_instruction_length: int | None
    average_instruction_length: float | None
    max_instruction_length: float | None
    unique_instructions: int | None

    # this is for datasets that do reranking
    num_top_ranked: int | None
    min_top_ranked_per_query: int | None
    average_top_ranked_per_query: float | None
    max_top_ranked_per_query: int | None


class SplitData(TypedDict, total=False):
    corpus: Mapping[str, str | dict[str, str]]
    queries: Mapping[str, str]
    relevant_docs: Mapping[str, Mapping[str, float]]
    instructions: Mapping[str, str] | None
    top_ranked: Mapping[str, list[str]] | None


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
    top_k = 100
    dataset: dict[str, dict[str, SplitData]]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dataset = defaultdict(
            lambda: defaultdict(
                lambda: {
                    "corpus": {},
                    "queries": {},
                    "relevant_docs": {},
                    "instructions": None,
                    "top_ranked": None,
                }
            )
        )

    def convert_v1_dataset_format_to_v2(self):
        if not hasattr(self, "queries"):
            return
        self.dataset = defaultdict(
            lambda: defaultdict(
                lambda: {
                    "corpus": {},
                    "queries": {},
                    "relevant_docs": {},
                    "instructions": None,
                    "top_ranked": None,
                }
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
            corpus, queries, relevant_docs, instructions, top_ranked = (
                RetrievalDataLoader(
                    hf_repo=dataset_path,
                    revision=revision,
                    trust_remote_code=trust_remote_code,
                    split=split,
                    config=hf_subset,
                ).load()
            )

            self.dataset[hf_subset][split] = {
                "corpus": corpus,
                "queries": queries,
                "relevant_docs": relevant_docs,
                "instructions": instructions,
                "top_ranked": top_ranked,
            }

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

        ndcg, _map, recall, precision, naucs, task_scores = retriever.evaluate(
            data_split["relevant_docs"],
            results,
            retriever.k_values,
            ignore_identical_ids=self.ignore_identical_ids,
        )
        mrr, naucs_mrr = retriever.evaluate_custom(
            data_split["relevant_docs"], results, retriever.k_values
        )
        scores = make_score_dict(
            ndcg, _map, recall, precision, mrr, naucs, naucs_mrr, task_scores
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

        query_len = calculate_queries_length(queries)
        doc_len = calculate_corpus_length(corpus)
        num_documents = len(doc_len) if corpus is not None else 0
        num_queries = len(query_len)
        num_relevant_docs = sum(len(relevant_docs[qid]) for qid in relevant_docs)
        none_queries = sum(q is None or len(q) == 0 for q in queries.values())

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

        if instructions is not None and len(instructions) > 0:
            instructions_len = [
                len(instruction) for instruction in instructions.values()
            ]
            num_instructions = len(instructions)
            average_instruction_length = sum(instructions_len)
            min_instruction_length = min(instructions_len)
            max_instruction_length = max(instructions_len)
            unique_instructions = len(set(instructions))
        else:
            num_instructions = None
            average_instruction_length = None
            min_instruction_length = None
            max_instruction_length = None
            unique_instructions = None

        if top_ranked is not None and num_queries and len(top_ranked) > 0:
            top_ranked_per_query = [len(docs) for docs in top_ranked.values()]
            num_top_ranked = len(top_ranked_per_query)
            min_top_ranked_per_query = min(top_ranked_per_query)
            average_top_ranked_per_query = sum(top_ranked_per_query) / num_queries
            max_top_ranked_per_query = max(top_ranked_per_query)
        else:
            num_top_ranked = None
            min_top_ranked_per_query = None
            average_top_ranked_per_query = None
            max_top_ranked_per_query = None

        return RetrievalDescriptiveStatistics(
            num_samples=num_documents + num_queries,
            number_of_characters=sum(query_len) + sum(doc_len),
            # documents
            num_documents=num_documents,
            min_document_length=min(doc_len),
            average_document_length=sum(doc_len) / num_documents,
            max_document_length=max(doc_len),
            unique_documents=len(set(corpus)),
            # queries
            num_queries=num_queries,
            min_query_length=min(query_len),
            average_query_length=sum(query_len) / num_queries,
            max_query_length=max(query_len),
            unique_queries=len(set(queries)),
            none_queries=none_queries,
            # relevant docs
            num_relevant_docs=num_relevant_docs,
            min_relevant_docs_per_query=min(qrels_lengths),
            average_relevant_docs_per_query=qrels_per_doc,
            max_relevant_docs_per_query=max(qrels_lengths),
            unique_relevant_docs=unique_qrels,
            # instructions
            num_instructions=num_instructions,
            min_instruction_length=min_instruction_length,
            average_instruction_length=average_instruction_length,
            max_instruction_length=max_instruction_length,
            unique_instructions=unique_instructions,
            # top ranked
            num_top_ranked=num_top_ranked,
            min_top_ranked_per_query=min_top_ranked_per_query,
            average_top_ranked_per_query=average_top_ranked_per_query,
            max_top_ranked_per_query=max_top_ranked_per_query,
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
            suffix: str,
            converter: Callable[[Any, Any], dict[str, Any]],
        ) -> None:
            sections = {}
            for split, items in data.items():
                sections[split] = Dataset.from_list(
                    [converter(idx, item) for idx, item in items.items()]
                )
            DatasetDict(sections).push_to_hub(repo_name, suffix)

        if self.metadata.is_multilingual:
            for lang in self.dataset["queries"]:
                logger.info(f"Converting {lang} of {self.metadata.name}")
                push_section(
                    self.dataset["queries"][lang],
                    f"{lang}-queries",
                    lambda idx, text: {"_id": idx, "text": text},
                )
                push_section(
                    self.dataset["corpus"][lang],
                    f"{lang}-corpus",
                    lambda idx, text: {
                        "_id": idx,
                        "text": format_text_field(text),
                        "title": "",
                    },
                )
                # Handle relevant_docs separately since one entry expands to multiple records.
                relevant_sections = {}
                for split, queries in self.dataset["relevant_docs"][lang].items():
                    entries = []
                    for query_id, docs in queries.items():
                        for doc_id, score in docs.items():
                            entries.append(
                                {
                                    "query-id": query_id,
                                    "corpus-id": doc_id,
                                    "score": score,
                                }
                            )
                    relevant_sections[split] = Dataset.from_list(entries)
                DatasetDict(relevant_sections).push_to_hub(repo_name, f"{lang}-qrels")

                if self.dataset["instructions"]:
                    push_section(
                        self.dataset["instructions"][lang],
                        f"{lang}-instruction",
                        lambda idx, text: {"query-id": idx, "instruction": text},
                    )
                if self.dataset["top_ranked"]:
                    push_section(
                        self.dataset["top_ranked"][lang],
                        f"{lang}-top_ranked",
                        lambda idx, docs: {"query-id": idx, "corpus-ids": docs},
                    )
        else:
            # For non-multilingual cases, flatten the structure if a "default" key exists.
            if "default" in self.dataset["queries"]:
                self.dataset["queries"] = self.dataset["queries"]["default"]
                self.dataset["corpus"] = self.dataset["corpus"]["default"]
                self.dataset["relevant_docs"] = self.dataset["relevant_docs"]["default"]
                if self.dataset["instructions"]:
                    self.dataset["instructions"] = self.dataset["instructions"][
                        "default"
                    ]
                if self.dataset["top_ranked"]:
                    self.dataset["top_ranked"] = self.dataset["top_ranked"]["default"]

            push_section(
                self.dataset["queries"],
                "queries",
                lambda idx, text: {"_id": idx, "text": text},
            )
            push_section(
                self.dataset["corpus"],
                "corpus",
                lambda idx, text: {
                    "_id": idx,
                    "text": format_text_field(text),
                    "title": text.get("title", "") if isinstance(text, dict) else "",
                },
            )
            # Process relevant_docs with flattening.
            relevant_sections = {}
            for split, queries in self.dataset["relevant_docs"].items():
                entries = []
                for query_id, docs in queries.items():
                    for doc_id, score in docs.items():
                        entries.append(
                            {"query-id": query_id, "corpus-id": doc_id, "score": score}
                        )
                relevant_sections[split] = Dataset.from_list(entries)
            DatasetDict(relevant_sections).push_to_hub(repo_name, "default")

            if self.dataset["instructions"]:
                push_section(
                    self.dataset["instructions"],
                    "instruction",
                    lambda idx, text: {"query-id": idx, "instruction": text},
                )
            if self.dataset["top_ranked"]:
                push_section(
                    self.dataset["top_ranked"],
                    "top_ranked",
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
