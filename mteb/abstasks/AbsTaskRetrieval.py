from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from time import time
from typing import Any

from datasets import Dataset, DatasetDict

from mteb.abstasks.TaskMetadata import HFSubset

from ..evaluation.evaluators import RetrievalEvaluator
from ..evaluation.evaluators.utils import make_score_dict
from ..load_results.task_results import ScoresDict
from .AbsTask import AbsTask
from .dataloaders import RetrievalDataLoader
from .TaskMetadata import DescriptiveStatistics

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


class AbsTaskRetrieval(AbsTask):
    """Abstract class for retrieval experiments.

    Child-classes must implement the following properties:

    self.corpus: dict[str, dict[str, str]]
        Semantically, it should contain dict[split_name, dict[sample_id, dict[str, str]]]
        E.g. {"test": {"document_one": {"_id": "d1", "title": "title", "text": "text"}}}

    self.queries: dict[str, dict[str, Union[str, list[str]]]]
        Semantically, it should contain dict[split_name, dict[sample_id, str]] or dict[split_name, dict[sample_id, list[str]]] for conversations
        E.g. {"test": {"q1": "query"}}
        or {"test": {"q1": ["turn1", "turn2", "turn3"]}}

    self.relevant_docs: dict[str, dict[str, dict[str, int]]]
        Semantically, it should contain dict[split_name, dict[sample_id, dict[doc_id, score]]]
        E.g.: {"test": {"q1": {"document_one": 1}}}

    Child classes may optionally implement the following properties (top_ranked for reranking and instructions if needed):

    self.top_ranked: dict[str, dict[str, list[str]]] or dict[str, dict[str, dict[str, float]]]
        Semantically, it should contain dict[split_name, dict[sample_id, list[doc_id]]] or dict[split_name, dict[sample_id, dict[doc_id, score]]]
        E.g.: {"test": {"q1": ["document_one", "document_two"]}} or {"test": {"q1": {"document_one": 1, "document_two": 0.5}}}

    self.instructions: dict[str, dict[str, str]] or dict[str, dict[str, list[str]]]
        Semantically, it should contain dict[split_name, dict[sample_id, str]]. If there are multiple instructions per query, please duplicate the queries and give them unique ids for consolidation.
        E.g. {"test": {"query-id1": "instruction text"}}
    """

    ignore_identical_ids: bool = False
    abstask_prompt = "Retrieve text based on user query."
    instructions = None
    top_ranked = None

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus = defaultdict(dict)
        self.queries = defaultdict(dict)
        self.relevant_docs = defaultdict(dict)
        self.instructions = None
        self.top_ranked = None

        dataset_path = self.metadata.dataset["path"]
        eval_splits = kwargs.get("eval_splits", self.metadata.eval_splits)
        trust_remote_code = self.metadata.dataset.get("trust_remote_code", False)

        def process_data(split: str, lang: str | None = None):
            """Helper function to load and process data for a given split and language"""
            corpus, queries, qrels, instructions, top_ranked = RetrievalDataLoader(
                hf_repo=dataset_path,
                trust_remote_code=trust_remote_code,
                split=split,
                config=lang,
            ).load()

            if lang:
                self.corpus[lang][split] = corpus
                self.queries[lang][split] = queries
                self.relevant_docs[lang][split] = qrels
            else:
                self.corpus[split] = corpus
                self.queries[split] = queries
                self.relevant_docs[split] = qrels

            if instructions:
                if self.instructions is None:
                    self.instructions = defaultdict(dict)
                if lang:
                    self.instructions[lang][split] = instructions
                else:
                    self.instructions[split] = instructions

            if top_ranked:
                if self.top_ranked is None:
                    self.top_ranked = defaultdict(dict)
                if lang:
                    self.top_ranked[lang][split] = top_ranked
                else:
                    self.top_ranked[split] = top_ranked

        if self.is_multilingual:
            for lang in self.metadata.eval_langs:
                for split in eval_splits:
                    process_data(split, lang)
        else:
            for split in eval_splits:
                process_data(split)
        self.data_loaded = True

    def evaluate(
        self,
        model,
        split: str = "test",
        subsets_to_run: list[HFSubset] | None = None,
        *,
        encode_kwargs: dict[str, Any] = {},
        **kwargs,
    ) -> dict[HFSubset, ScoresDict]:
        retriever = RetrievalEvaluator(
            retriever=model,
            task_name=self.metadata.name,
            encode_kwargs=encode_kwargs,
            **kwargs,
        )

        scores = {}
        hf_subsets = list(self.hf_subsets) if self.is_multilingual else ["default"]
        if subsets_to_run is not None:
            hf_subsets = [s for s in hf_subsets if s in subsets_to_run]

        for hf_subset in hf_subsets:
            logger.info(f"Subset: {hf_subset}")

            if hf_subset == "default" and "default" not in self.corpus:
                corpus = self.corpus[split]
                queries = self.queries[split]
                relevant_docs = self.relevant_docs[split]
                top_ranked = self.top_ranked[split] if self.top_ranked else None
                instructions = self.instructions[split] if self.instructions else None
            else:
                corpus = self.corpus[hf_subset][split]
                queries = self.queries[hf_subset][split]
                relevant_docs = self.relevant_docs[hf_subset][split]
                top_ranked = (
                    self.top_ranked[hf_subset][split] if self.top_ranked else None
                )
                instructions = (
                    self.instructions[hf_subset][split] if self.instructions else None
                )

            scores[hf_subset] = self._evaluate_subset(
                retriever,
                corpus,
                queries,
                relevant_docs,
                hf_subset,
                top_ranked,
                instructions,
                **kwargs,
            )
        return scores

    def _evaluate_subset(
        self,
        retriever: RetrievalEvaluator,
        corpus: dict[str, dict[str, str]],
        queries: dict[str, str],
        relevant_docs: dict[str, dict[str, int]],
        hf_subset: str,
        top_ranked: dict[str, list[str]] | None = None,
        instructions: dict[str, str] | None = None,
        save_predictions: bool = False,
        export_errors: bool = False,
        save_qrels: bool = False,
        output_folder: str = "results",
        results: dict[str, dict[str, float]] | None = None,
        top_k: int | None = None,
        **kwargs,
    ) -> ScoresDict:
        """Evaluate the retrieval task for a given subset of the dataset.

        Args:
            retriever: Evaluation object
            corpus: Corpus to evaluate on
            queries: Queries to evaluate on
            relevant_docs: Relevant documents for the queries
            hf_subset: Subset of the dataset
            top_ranked: Top ranked documents (used for reranking)
            instructions: Instructions for the queries (used for InstructRetrieval/Reranking)
            save_predictions: Whether to save the predictions
            export_errors: Whether to export errors
            save_qrels: Whether to save the qrels
            output_folder: Folder to save the results
            results: Results from retrieval from previous run
            top_k: Top k documents to consider
            **kwargs: kwargs

        Returns:
            ScoresDict: Evaluation scores
        """
        if not results:
            # perform the retrieval here
            start_time = time()
            results = retriever(
                corpus,
                queries,
                instructions=instructions,
                top_ranked=top_ranked,
                **kwargs,
            )
            end_time = time()
            logger.info(f"Time taken to retrieve: {end_time - start_time:.2f} seconds")

        if save_predictions or export_errors or save_qrels:
            output_folder = Path(output_folder)
            if not os.path.isdir(output_folder):
                os.makedirs(output_folder)

        if save_predictions:
            if top_k is not None:
                for qid in list(results.keys()):
                    doc_ids = set(
                        sorted(
                            results[qid], key=lambda x: results[qid][x], reverse=True
                        )[:top_k]
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
                json.dump(relevant_docs, f)

        ndcg, _map, recall, precision, naucs, task_scores = retriever.evaluate(
            relevant_docs,
            results,
            retriever.k_values,
            ignore_identical_ids=self.ignore_identical_ids,
            task_name=self.metadata.name,
        )

        mrr, naucs_mrr = retriever.evaluate_custom(
            relevant_docs, results, retriever.k_values, "mrr"
        )
        scores = make_score_dict(
            ndcg, _map, recall, precision, mrr, naucs, naucs_mrr, task_scores
        )
        self._add_main_score(scores)

        if export_errors:
            errors = {}

            top_k = top_k or 1
            if not save_predictions and top_k == 1:
                for qid in results.keys():
                    doc_scores = results[qid]
                    sorted_docs = sorted(
                        doc_scores.items(), key=lambda x: x[1], reverse=True
                    )[:top_k]
                    results[qid] = dict(sorted_docs)
            for qid, retrieved_docs in results.items():
                expected_docs = relevant_docs[qid]
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
            with open(errors_save_path, "w") as f:
                json.dump(errors, f)

        return scores

    def _add_main_score(self, scores: ScoresDict) -> None:
        scores["main_score"] = scores[self.metadata.main_score]

    def _calculate_metrics_from_split(
        self, split: str, hf_subset: str | None = None, compute_overall: bool = False
    ) -> RetrievalDescriptiveStatistics:
        top_ranked = None
        instructions = None
        if hf_subset and hf_subset in self.queries:
            # BrightRetrieval has different splits for different subsets of the corpus.
            if (
                self.corpus.get(hf_subset, None) is None
                or self.corpus[hf_subset].get(split, None) is None
            ):
                return {}

            queries = self.queries[hf_subset][split]
            corpus = self.corpus[hf_subset][split]
            relevant_docs = self.relevant_docs[hf_subset][split]
            if self.instructions is not None:
                instructions = self.instructions[hf_subset][split]
            if self.top_ranked is not None:
                top_ranked = self.top_ranked[hf_subset][split]
        elif compute_overall:
            queries = {}
            corpus = {}
            relevant_docs = {}
            instructions = {}
            top_ranked = {}
            for hf_subset in self.metadata.eval_langs:
                # BrightRetrieval has different splits for different subsets of the corpus.
                if (
                    self.corpus.get(hf_subset, None) is None
                    or self.corpus[hf_subset].get(split, None) is None
                ):
                    continue
                queries.update(process_docs(self.queries, hf_subset, split))
                corpus.update(process_docs(self.corpus, hf_subset, split))
                relevant_docs.update(
                    process_relevant_docs(self.relevant_docs, hf_subset, split)
                )
                if self.instructions is not None:
                    instructions.update(
                        process_docs(self.instructions, hf_subset, split)
                    )
                if self.top_ranked is not None:
                    top_ranked.update(process_docs(self.top_ranked, hf_subset, split))
        else:
            if "default" in self.queries and split != "default":
                return self._calculate_metrics_from_split(
                    split=split, hf_subset="default"
                )
            queries = self.queries[split]
            corpus = self.corpus[split]
            relevant_docs = self.relevant_docs[split]
            if self.instructions is not None:
                instructions = self.instructions[split]
            if self.top_ranked is not None:
                top_ranked = self.top_ranked[split]

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

        if self.instructions is not None:
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

        if self.top_ranked is not None and num_queries:
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
        def format_text_field(text):
            """Formats the text field to match loader expectations."""
            if isinstance(text, str):
                return text
            return f"{text.get('title', '')} {text.get('text', '')}".strip()

        if self.is_multilingual:
            for config in self.queries:
                logger.info(f"Converting {config} of {self.metadata.name}")

                queries_dataset = {}
                for split in self.queries[config]:
                    queries_dataset[split] = Dataset.from_list(
                        [
                            {
                                "_id": idx,
                                "text": text,
                            }
                            for idx, text in self.queries[config][split].items()
                        ]
                    )
                queries_dataset = DatasetDict(queries_dataset)
                queries_dataset.push_to_hub(repo_name, f"{config}-queries")

                corpus_dataset = {}
                for split in self.corpus[config]:
                    corpus_dataset[split] = Dataset.from_list(
                        [
                            {
                                "_id": idx,
                                "text": format_text_field(text),
                                "title": text.get("title", "")
                                if isinstance(text, dict)
                                else "",
                            }
                            for idx, text in self.corpus[config][split].items()
                        ]
                    )

                corpus_dataset = DatasetDict(corpus_dataset)
                corpus_dataset.push_to_hub(repo_name, f"{config}-corpus")

                relevant_docs_dataset = {}
                for split in self.relevant_docs[config]:
                    relevant_docs_dataset[split] = Dataset.from_list(
                        [
                            {"query-id": query_id, "corpus-id": doc_id, "score": score}
                            for query_id, docs in self.relevant_docs[config][
                                split
                            ].items()
                            for doc_id, score in docs.items()
                        ]
                    )
                relevant_docs_dataset = DatasetDict(relevant_docs_dataset)
                relevant_docs_dataset.push_to_hub(repo_name, f"{config}-qrels")

                if self.instructions:
                    instructions_dataset = {}
                    for split in self.instructions[config]:
                        instructions_dataset[split] = Dataset.from_list(
                            [
                                {
                                    "query-id": idx,
                                    "instruction": text,
                                }
                                for idx, text in self.instructions[config][
                                    split
                                ].items()
                            ]
                        )
                    instructions_dataset = DatasetDict(instructions_dataset)
                    instructions_dataset.push_to_hub(repo_name, f"{config}-instruction")
                if self.top_ranked:
                    top_ranked_dataset = {}
                    for split in self.top_ranked[config]:
                        top_ranked_dataset[split] = Dataset.from_list(
                            [
                                {
                                    "query-id": query_id,
                                    "corpus-ids": docs,
                                }
                                for query_id, docs in self.top_ranked[config][
                                    split
                                ].items()
                            ]
                        )
                    top_ranked_dataset = DatasetDict(top_ranked_dataset)
                    top_ranked_dataset.push_to_hub(repo_name, f"{config}-top_ranked")
        else:
            if "default" in self.queries:
                # old rerankers have additional default split
                self.queries = self.queries["default"]
                self.corpus = self.corpus["default"]
                self.relevant_docs = self.relevant_docs["default"]
                if self.instructions:
                    self.instructions = self.instructions["default"]
                if self.top_ranked:
                    self.top_ranked = self.top_ranked["default"]

            queries_dataset = {}
            for split in self.queries:
                queries_dataset[split] = Dataset.from_list(
                    [
                        {
                            "_id": idx,
                            "text": text,
                        }
                        for idx, text in self.queries[split].items()
                    ]
                )
            queries_dataset = DatasetDict(queries_dataset)
            queries_dataset.push_to_hub(repo_name, "queries")
            corpus_dataset = {}
            for split in self.corpus:
                corpus_dataset[split] = Dataset.from_list(
                    [
                        {
                            "_id": idx,
                            "text": format_text_field(text),
                            "title": text.get("title", "")
                            if isinstance(text, dict)
                            else "",
                        }
                        for idx, text in self.corpus[split].items()
                    ]
                )

            corpus_dataset = DatasetDict(corpus_dataset)
            corpus_dataset.push_to_hub(repo_name, "corpus")
            relevant_docs_dataset = {}
            for split in self.relevant_docs:
                relevant_docs_dataset[split] = Dataset.from_list(
                    [
                        {"query-id": query_id, "corpus-id": doc_id, "score": score}
                        for query_id, docs in self.relevant_docs[split].items()
                        for doc_id, score in docs.items()
                    ]
                )
            relevant_docs_dataset = DatasetDict(relevant_docs_dataset)
            relevant_docs_dataset.push_to_hub(repo_name, "default")
            if self.instructions:
                instructions_dataset = {}
                for split in self.instructions:
                    instructions_dataset[split] = Dataset.from_list(
                        [
                            {
                                "query-id": idx,
                                "instruction": text,
                            }
                            for idx, text in self.instructions[split].items()
                        ]
                    )
                instructions_dataset = DatasetDict(instructions_dataset)
                instructions_dataset.push_to_hub(repo_name, "instruction")
            if self.top_ranked:
                top_ranked_dataset = {}
                for split in self.top_ranked:
                    top_ranked_dataset[split] = Dataset.from_list(
                        [
                            {
                                "query-id": query_id,
                                "corpus-ids": docs,
                            }
                            for query_id, docs in self.top_ranked[split].items()
                        ]
                    )
                top_ranked_dataset = DatasetDict(top_ranked_dataset)
                top_ranked_dataset.push_to_hub(repo_name, "top_ranked")


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
    collection: dict[str, dict[str, dict[str, str] | str]], hf_subset: str, split: str
) -> dict[str, str]:
    """Collections can contain overlapping ids in different splits. Prepend split to avoid this"""
    return {
        f"{split}_{hf_subset}_{k}": v for k, v in collection[hf_subset][split].items()
    }


def process_relevant_docs(
    collection: dict[str, dict[str, dict[str, dict[str, int]]]],
    hf_subset: str,
    split: str,
) -> dict[str, dict[str, int]]:
    """Collections can contain overlapping ids in different splits. Prepend split to avoid this"""
    return_collection = {}
    for query_id, relevant in collection[hf_subset][split].items():
        return_collection[f"{split}_{hf_subset}_{query_id}"] = {
            f"{split}_{hf_subset}_{doc_id}": value for doc_id, value in relevant.items()
        }
    return return_collection
