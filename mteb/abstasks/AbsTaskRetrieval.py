from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from time import time
from typing import Any

from mteb.abstasks.TaskMetadata import HFSubset

from ..evaluation.evaluators import RetrievalEvaluator
from ..load_results.task_results import ScoresDict
from .AbsTask import AbsTask, DescriptiveStatistics
from .dataloaders import HFDataLoader

logger = logging.getLogger(__name__)


class RetrievalDescriptiveStatistics(DescriptiveStatistics):
    """Descriptive statistics for Retrieval

    Attributes:
        num_queries: number of samples in the dataset
        average_document_length: Average length of documents
        average_query_length: Average length of queries
        num_documents: Number of documents
        average_relevant_docs_per_query: Average number of relevant documents per query
        average_instruction_length: Average length of instructions
        average_num_instructions_per_query: Average number of instructions per query
        average_top_ranked_per_query: Average number of top ranked documents per query
    """

    num_queries: int
    average_document_length: float
    average_query_length: float
    num_documents: int
    average_relevant_docs_per_query: float
    # these are for datasets with instructions
    average_instruction_length: float
    average_num_instructions_per_query: float
    # this is for datasets that do reranking
    average_top_ranked_per_query: float


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
        Semantically, it should contain dict[split_name, dict[sample_id, str]] or dict[split_name, dict[sample_id, list[str]]] for multiple instructions per query
        E.g. {"test": {"q1": "instruction"}} or {"test": {"q1": ["instruction1", "instruction2"]}}
    """

    ignore_identical_ids: bool = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load_data(self, **kwargs):
        if self.data_loaded:
            return
        self.corpus, self.queries, self.relevant_docs = {}, {}, {}
        self.instructions, self.top_ranked = None, None
        dataset_path = self.metadata_dict["dataset"]["path"]
        hf_repo_qrels = (
            dataset_path + "-qrels" if "clarin-knext" in dataset_path else None
        )
        for split in kwargs.get("eval_splits", self.metadata_dict["eval_splits"]):
            corpus, queries, qrels, instructions, top_ranked = HFDataLoader(
                hf_repo=dataset_path,
                hf_repo_qrels=hf_repo_qrels,
                streaming=False,
                keep_in_memory=False,
            ).load(split=split)
            # Conversion from DataSet
            queries = {query["id"]: query["text"] for query in queries}
            corpus = {
                doc["id"]: doc.get("title", "") + " " + doc["text"] for doc in corpus
            }
            self.corpus[split], self.queries[split], self.relevant_docs[split] = (
                corpus,
                queries,
                qrels,
            )

            # optional args
            if instructions:
                self.instructions = {
                    split: {inst["query"]: inst["instruction"] for inst in instructions}
                }
            if top_ranked:
                self.top_ranked = {
                    split: {tr["query-id"]: tr["corpus-ids"] for tr in top_ranked}
                }

        self.data_loaded = True

    def evaluate(
        self,
        model,
        split: str = "test",
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

        for hf_subset in hf_subsets:
            logger.info(f"Subset: {hf_subset}")

            if hf_subset == "default" and "default" not in self.corpus:
                corpus, queries, relevant_docs = (
                    self.corpus[split],
                    self.queries[split],
                    self.relevant_docs[split],
                )
                if self.top_ranked is not None:
                    kwargs["top_ranked"] = self.top_ranked[split]
                if self.instructions is not None:
                    kwargs["instructions"] = self.instructions[split]
            else:
                corpus, queries, relevant_docs = (
                    self.corpus[hf_subset][split],
                    self.queries[hf_subset][split],
                    self.relevant_docs[hf_subset][split],
                )
                if self.top_ranked is not None:
                    kwargs["top_ranked"] = self.top_ranked[hf_subset][split]
                if self.instructions is not None:
                    kwargs["instructions"] = self.instructions[hf_subset][split]

            scores[hf_subset] = self._evaluate_subset(
                retriever, corpus, queries, relevant_docs, hf_subset, **kwargs
            )
        return scores

    def _evaluate_subset(
        self, retriever, corpus, queries, relevant_docs, hf_subset: str, **kwargs
    ) -> ScoresDict:
        if "results" in kwargs:
            # reranking has already been done
            results = kwargs["results"]
        else:
            # perform the retrieval here
            start_time = time()
            # instructions can be a set of instructions for each query
            # TODO: add instructions here - in progress
            results = retriever(corpus, queries, **kwargs)
            end_time = time()
            logger.info(f"Time taken to retrieve: {end_time - start_time:.2f} seconds")

        save_predictions = kwargs.get("save_predictions", False)
        export_errors = kwargs.get("export_errors", False)
        if save_predictions or export_errors:
            output_folder = Path(kwargs.get("output_folder", "results"))
            if not os.path.isdir(output_folder):
                os.makedirs(output_folder)

        if save_predictions:
            top_k = kwargs.get("top_k", None)
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
        scores = {
            **{f"ndcg_at_{k.split('@')[1]}": v for (k, v) in ndcg.items()},
            **{f"map_at_{k.split('@')[1]}": v for (k, v) in _map.items()},
            **{f"recall_at_{k.split('@')[1]}": v for (k, v) in recall.items()},
            **{f"precision_at_{k.split('@')[1]}": v for (k, v) in precision.items()},
            **{f"mrr_at_{k.split('@')[1]}": v for (k, v) in mrr.items()},
            **{
                k.replace("@", "_at_").replace("_P", "_precision").lower(): v
                for k, v in naucs.items()
            },
            **{
                k.replace("@", "_at_").replace("_P", "_precision").lower(): v
                for k, v in naucs_mrr.items()
            },
            **task_scores,
        }
        self._add_main_score(scores)

        if export_errors:
            errors = {}

            top_k = kwargs.get("top_k", 1)
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
        if hf_subset:
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
            queries = self.queries[split]
            corpus = self.corpus[split]
            relevant_docs = self.relevant_docs[split]
            if self.instructions is not None:
                instructions = self.instructions[split]
            if self.top_ranked is not None:
                top_ranked = self.top_ranked[split]

        query_len, doc_len = calculate_length(queries, corpus)
        num_documents = len(corpus)
        num_queries = len(queries)

        # number of qrels that are not 0
        num_qrels_non_zero = sum(
            sum(1 for doc_id in docs if docs[doc_id] != 0)
            for docs in relevant_docs.values()
        )
        qrels_per_doc = num_qrels_non_zero / len(relevant_docs) if num_queries else 0

        if self.instructions is not None:
            total_instructions_len = sum(
                [
                    len(instruction)
                    if isinstance(instruction, str)
                    else sum([len(i) for i in instruction])
                    for instruction in instructions.values()
                ]
            )
            # flatten the values in instructions to get the number of them
            num_instructions = sum(
                1 if isinstance(instruction, str) else len(instruction)
                for instruction in instructions.values()
            )
        else:
            total_instructions_len = 0
            num_instructions = 0

        if self.top_ranked is not None:
            top_ranked_per_query = (
                sum(len(docs) for docs in top_ranked.values()) / num_queries
                if num_queries
                else 0
            )
        else:
            top_ranked_per_query = 0

        return RetrievalDescriptiveStatistics(
            average_document_length=doc_len,
            average_query_length=query_len,
            num_documents=num_documents,
            num_queries=num_queries,
            average_relevant_docs_per_query=qrels_per_doc,
            average_instruction_length=total_instructions_len / num_instructions
            if num_instructions
            else 0,
            average_num_instructions_per_query=num_instructions / num_queries
            if num_queries
            else 0,
            average_top_ranked_per_query=top_ranked_per_query,
        )


def calculate_length(
    queries: dict[str, str], corpus: dict[str, str]
) -> tuple[float, float]:
    queries_lens = []
    doc_lens = []
    for query in queries.values():
        queries_lens.append(len(query))

    for doc in corpus.values():
        doc_lens.append(len(doc))

    doc_len = sum(doc_lens) / len(doc_lens) if doc_lens else 0
    query_len = sum(queries_lens) / len(queries_lens) if queries_lens else 0
    return query_len, doc_len


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
