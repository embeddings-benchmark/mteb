from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from time import time
from typing import Any

import tqdm
from datasets import Features, Value, load_dataset
from PIL import Image

from mteb.abstasks.AbsTask import AbsTask, ScoresDict

from ...evaluation.evaluators import Any2AnyMultiChoiceEvaluator
from ..TaskMetadata import DescriptiveStatistics

logger = logging.getLogger(__name__)


class HFDataLoader:
    def __init__(
        self,
        hf_repo: str | None = None,
        hf_repo_qrels: str | None = None,
        data_folder: str | None = None,
        prefix: str | None = None,
        corpus_file: str = "corpus.jsonl",
        query_file: str = "queries.jsonl",
        qrels_folder: str = "qrels",
        qrels_file: str = "",
        streaming: bool = False,
        keep_in_memory: bool = False,
    ):
        self.corpus = {}
        self.queries = {}
        self.qrels = {}
        self.hf_repo = hf_repo
        if hf_repo:
            # By default fetch qrels from same repo not a second repo with "-qrels" like in original
            self.hf_repo_qrels = hf_repo_qrels if hf_repo_qrels else hf_repo
        else:
            # data folder would contain these files:
            # (1) fiqa/corpus.jsonl  (format: jsonlines)
            # (2) fiqa/queries.jsonl (format: jsonlines)
            # (3) fiqa/qrels/test.tsv (format: tsv ("\t"))
            if prefix:
                query_file = prefix + "-" + query_file
                qrels_folder = prefix + "-" + qrels_folder

            self.corpus_file = (
                os.path.join(data_folder, corpus_file) if data_folder else corpus_file
            )
            self.query_file = (
                os.path.join(data_folder, query_file) if data_folder else query_file
            )
            self.qrels_folder = (
                os.path.join(data_folder, qrels_folder) if data_folder else None
            )
            self.qrels_file = qrels_file
        self.streaming = streaming
        self.keep_in_memory = keep_in_memory

    @staticmethod
    def check(fIn: str, ext: str):
        if not os.path.exists(fIn):
            raise ValueError(f"File {fIn} not present! Please provide accurate file.")

        if not fIn.endswith(ext):
            raise ValueError(f"File {fIn} must be present with extension {ext}")

    def load(
        self, split="test"
    ) -> tuple[
        dict[str, dict[str, str | Image.Image]],
        dict[str, dict[str, str | Image.Image]],
        dict[str, dict[str, int]],
    ]:
        if not self.hf_repo:
            self.qrels_file = os.path.join(self.qrels_folder, split + ".tsv")
            self.check(fIn=self.corpus_file, ext="jsonl")
            self.check(fIn=self.query_file, ext="jsonl")
            self.check(fIn=self.qrels_file, ext="tsv")

        if not len(self.corpus):
            logger.info("Loading Corpus...")
            self._load_corpus()
            logger.info("Loaded %d %s Documents.", len(self.corpus), split.upper())
            logger.info("Doc Example: %s", self.corpus[0])

        if not len(self.queries):
            logger.info("Loading Queries...")
            self._load_queries(split)

        self._load_qrels(split)
        # filter queries with no qrels
        qrels_dict = defaultdict(dict)

        def qrels_dict_init(row):
            qrels_dict[row["query-id"]][row["corpus-id"]] = int(row["score"])

        self.qrels.map(qrels_dict_init)
        self.qrels = qrels_dict
        self.queries = self.queries.filter(lambda x: x["id"] in self.qrels)
        logger.info("Loaded %d %s Queries.", len(self.queries), split.upper())
        logger.info("Query Example: %s", self.queries[0])

        return self.corpus, self.queries, self.qrels

    def load_corpus(self) -> dict[str, dict[str, str]]:
        if not self.hf_repo:
            self.check(fIn=self.corpus_file, ext="jsonl")

        if not len(self.corpus):
            logger.info("Loading Corpus...")
            self._load_corpus()
            logger.info("Loaded %d %s Documents.", len(self.corpus))
            logger.info("Doc Example: %s", self.corpus[0])

        return self.corpus

    def _load_corpus(self):
        if self.hf_repo:
            corpus_ds = load_dataset(
                self.hf_repo,
                "corpus",
                keep_in_memory=self.keep_in_memory,
                streaming=self.streaming,
            )["corpus"]
        else:
            corpus_ds = load_dataset(
                "json",
                data_files=self.corpus_file,
                streaming=self.streaming,
                keep_in_memory=self.keep_in_memory,
            )
        self.corpus = corpus_ds

    def _load_queries(self, split):
        if self.hf_repo:
            queries_ds = load_dataset(
                self.hf_repo,
                "query",
                keep_in_memory=self.keep_in_memory,
                streaming=self.streaming,
            )[split]
        else:
            queries_ds = load_dataset(
                "json",
                data_files=self.query_file,
                streaming=self.streaming,
                keep_in_memory=self.keep_in_memory,
            )
        self.queries = queries_ds

    def _load_qrels(self, split):
        if self.hf_repo:
            qrels_ds = load_dataset(
                self.hf_repo_qrels,
                "qrels",
                keep_in_memory=self.keep_in_memory,
                streaming=self.streaming,
            )[split]
        else:
            qrels_ds = load_dataset(
                "csv",
                data_files=self.qrels_file,
                delimiter="\t",
                keep_in_memory=self.keep_in_memory,
            )

        if "Q0" in qrels_ds.column_names:
            qrels_ds = qrels_ds.remove_columns("Q0")
        features = Features(
            {
                "query-id": Value("string"),
                "corpus-id": Value("string"),
                "score": Value("float"),
            }
        )
        # Some datasets may have extra columns, e.g. `difficulty` in qrels for FORB.
        qrels_ds = qrels_ds.select_columns(["query-id", "corpus-id", "score"]).cast(
            features
        )
        self.qrels = qrels_ds


class Any2AnyMutipleChoiceDescriptiveStatistics(DescriptiveStatistics):
    """Descriptive statistics for Any2TextMutipleChoice

    Attributes:
        num_samples: Number of queries and documents
        num_queries: number of queries in the dataset
        num_documents: Number of documents
        number_of_characters: Total number of text characters in the dataset

        For text only:
        min_document_length: Minimum length of documents
        average_document_length: Average length of documents
        max_document_length: Maximum length of documents
        unique_documents: Number of unique documents

        For text only:
        min_query_length: Minimum length of queries
        average_query_length: Average length of queries
        max_query_length: Maximum length of queries
        unique_queries: Number of unique queries

        For images:
        num_query_images: Number of query images
        num_document_images: Number of document images

        For images:
        min_document_image_width: Minimum width of document images
        average_document_image_width: Average width of document images
        max_document_image_width: Maximum width of document images
        min_document_image_height: Minimum height of document images
        average_document_image_height: Average height of document images
        max_document_image_height: Maximum height of document images

        For images:
        min_query_image_width: Minimum width of query images
        average_query_image_width: Average width of query images
        max_query_image_width: Maximum width of query images
        min_query_image_height: Minimum height of query images
        average_query_image_height: Average height of query images
        max_query_image_height: Maximum height of query images

        min_relevant_docs_per_query: Minimum number of relevant documents per query
        average_relevant_docs_per_query: Average number of relevant documents per query
        max_relevant_docs_per_query: Maximum number of relevant documents per query
        unique_relevant_docs: Number of unique relevant documents

        min_irrelevant_docs_per_query: Minimum number of irrelevant documents per query
        average_irrelevant_docs_per_query: Average number of irrelevant documents per query
        max_irrelevant_docs_per_query: Maximum number of irrelevant documents per query
        unique_irrelevant_docs: Number of unique irrelevant documents
    """

    num_samples: int
    num_queries: int
    num_documents: int
    number_of_characters: int

    min_document_length: int
    average_document_length: float
    max_document_length: int
    unique_documents: int
    num_document_images: int

    min_document_image_width: float
    average_document_image_width: float
    max_document_image_width: float
    min_document_image_height: float
    average_document_image_height: float
    max_document_image_height: float

    min_query_length: int
    average_query_length: float
    max_query_length: int
    unique_queries: int
    num_query_images: int

    min_query_image_width: float
    average_query_image_width: float
    max_query_image_width: float
    min_query_image_height: float
    average_query_image_height: float
    max_query_image_height: float

    min_relevant_docs_per_query: int
    average_relevant_docs_per_query: float
    max_relevant_docs_per_query: int
    unique_relevant_docs: int


class AbsTaskAny2AnyMultiChoice(AbsTask):
    """Abstract class for Any2Any multiple choice experiments

    This is NOT a retrieval task: there is one correct answer among a set of candidates, which are a subset of the corpus, indicated in qrels with a relevance of 0

    Child-classes must implement the following properties:

    self.corpus: dict[str, dict[str, str]]
        Semantically, it should contain dict[split_name, dict[sample_id, dict[str, str]]]
        E.g. {"test": {"document_one": {"_id": "d1", "title": "title", "text": "text"}}}

    self.queries: dict[str, dict[str, Union[str, List[str]]]]
        Semantically, it should contain dict[split_name, dict[sample_id, str]] or dict[split_name, dict[sample_id, List[str]]] for conversations
        E.g. {"test": {"q1": "query"}}
        or {"test": {"q1": ["turn1", "turn2", "turn3"]}}

    self.relevant_docs: dict[str, dict[str, dict[str, int]]]
        Semantically, it should contain dict[split_name, dict[sample_id, dict[doc_id, score]]]
        E.g.: {"test": {"q1": {"document_one": 1}}} for hard positive samples (the correct choice)
        E.g.: {"test": {"q1": {"document_two": 0}}} for hard negative samples (incorrect choices from the same query)
    """

    ignore_identical_ids: bool = False
    skip_first_result: bool = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load_data(self, **kwargs):
        if self.data_loaded:
            return
        self.corpus, self.queries, self.relevant_docs = {}, {}, {}
        dataset_path = self.metadata_dict["dataset"]["path"]

        for split in kwargs.get("eval_splits", self.metadata_dict["eval_splits"]):
            corpus, queries, qrels = HFDataLoader(
                hf_repo=dataset_path,
                streaming=False,
                keep_in_memory=False,
            ).load(split=split)
            # directly pass in corpus and queries datasets now to prevent loading into memory
            # queries = {query["id"]: query for query in queries}
            # corpus = {doc["id"]: doc for doc in corpus}
            self.corpus[split], self.queries[split], self.relevant_docs[split] = (
                corpus,
                queries,
                qrels,
            )

        self.data_loaded = True

    def evaluate(
        self,
        model,
        split: str = "test",
        *,
        encode_kwargs: dict[str, Any] = {},
        **kwargs,
    ):
        retriever = Any2AnyMultiChoiceEvaluator(
            retriever=model,
            task_name=self.metadata.name,
            encode_kwargs=encode_kwargs,
            **kwargs,
        )

        scores = {}
        hf_subsets = list(self.hf_subsets) if self.is_multilingual else ["default"]

        for hf_subset in hf_subsets:
            logger.info(f"Subset: {hf_subset}")

            if hf_subset == "default":
                corpus, queries, relevant_docs = (
                    self.corpus[split],
                    self.queries[split],
                    self.relevant_docs[split],
                )
            else:
                corpus, queries, relevant_docs = (
                    self.corpus[hf_subset][split],
                    self.queries[hf_subset][split],
                    self.relevant_docs[hf_subset][split],
                )
            scores[hf_subset] = self._evaluate_subset(
                retriever, corpus, queries, relevant_docs, hf_subset, **kwargs
            )
        return scores

    def _evaluate_subset(
        self, retriever, corpus, queries, relevant_docs, hf_subset: str, **kwargs
    ):
        start_time = time()
        results = retriever(corpus, queries, relevant_docs)
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

        ndcg, _map, recall, precision, cv_recall, naucs = retriever.evaluate(
            relevant_docs,
            results,
            retriever.k_values,
            ignore_identical_ids=self.ignore_identical_ids,
            skip_first_result=self.skip_first_result,
        )
        mrr, naucs_mrr = retriever.evaluate_custom(
            relevant_docs, results, retriever.k_values, "mrr"
        )
        scores = {
            **{f"ndcg_at_{k.split('@')[1]}": v for (k, v) in ndcg.items()},
            **{f"map_at_{k.split('@')[1]}": v for (k, v) in _map.items()},
            **{f"recall_at_{k.split('@')[1]}": v for (k, v) in recall.items()},
            **{f"cv_recall_at_{k.split('@')[1]}": v for (k, v) in cv_recall.items()},
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
            "accuracy": recall["Recall@1"],
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
    ) -> Any2AnyMutipleChoiceDescriptiveStatistics:
        if hf_subset:
            queries = self.queries[hf_subset][split]
            corpus = self.corpus[hf_subset][split]
            relevant_docs = self.relevant_docs[hf_subset][split]
        elif compute_overall:
            queries = {}
            corpus = {}
            relevant_docs = {}
            for hf_subset in self.metadata.eval_langs:
                queries.update(process_docs(self.queries, hf_subset, split))
                corpus.update(process_docs(self.corpus, hf_subset, split))
                relevant_docs.update(
                    process_relevant_docs(self.relevant_docs, hf_subset, split)
                )
        else:
            queries = self.queries[split]
            corpus = self.corpus[split]
            relevant_docs = self.relevant_docs[split]

        queries_lens, doc_lens = [], []
        num_query_images = 0
        num_document_images = 0

        q_modality = queries[0]["modality"]
        unique_queries = len(set(queries["text"])) if "text" in q_modality else 0

        for query in tqdm.tqdm(queries, desc="queries:"):
            if "text" in q_modality:
                text_query = query["text"]
                queries_lens.append(len(text_query))
            if "image" in q_modality:
                num_query_images += 1

        d_modality = corpus[0]["modality"]
        unique_documents = len(set(corpus["text"])) if "text" in d_modality else 0

        for doc in tqdm.tqdm(corpus, desc="docs:"):
            if "text" in d_modality:
                text_doc = doc["text"]
                doc_lens.append(len(text_doc))
            if "image" in d_modality:
                num_document_images += 1

        total_doc_len = sum(doc_lens)
        total_query_len = sum(queries_lens)
        num_documents = len(corpus)
        num_queries = len(queries)

        d_modality = corpus[0]["modality"]
        imgs = [doc["image"] for doc in corpus if "image" in d_modality]
        d_img_widths, d_img_heights = [], []
        for img in imgs:
            width, height = img.size
            d_img_widths.append(height)
            d_img_heights.append(width)

        q_modality = queries[0]["modality"]
        imgs = [query["image"] for query in queries if "image" in q_modality]
        q_img_widths, q_img_heights = [], []
        for img in imgs:
            width, height = img.size
            q_img_widths.append(height)
            q_img_heights.append(width)

        # create a list of number of relevant docs per query
        queries_set = set(queries["id"])
        qrels_lengths = [
            len([v for k, v in relevant_docs[qid].items() if v != 0])
            for qid in tqdm.tqdm(relevant_docs.keys(), desc="qrels:")
            if qid in queries_set
        ]
        num_qrels = sum(qrels_lengths)
        qrels_per_doc = num_qrels / len(relevant_docs) if num_queries else 0
        unique_qrels = len({doc for qid in relevant_docs for doc in relevant_docs[qid]})

        return Any2AnyMutipleChoiceDescriptiveStatistics(
            number_of_characters=total_query_len + total_doc_len,
            num_samples=num_documents + num_queries,
            num_queries=num_queries,
            num_documents=num_documents,
            min_document_length=min(doc_lens) if doc_lens else 0,
            average_document_length=total_doc_len / len(doc_lens) if doc_lens else 0,
            max_document_length=max(doc_lens) if doc_lens else 0,
            unique_documents=unique_documents,
            min_document_image_width=min(d_img_widths) if d_img_widths else 0,
            average_document_image_width=sum(d_img_widths) / len(d_img_widths)
            if d_img_widths
            else 0,
            max_document_image_width=max(d_img_widths) if d_img_widths else 0,
            min_document_image_height=min(d_img_heights) if d_img_heights else 0,
            average_document_image_height=sum(d_img_heights) / len(d_img_heights)
            if d_img_heights
            else 0,
            max_document_image_height=max(d_img_heights) if d_img_heights else 0,
            num_document_images=num_document_images,
            min_query_length=min(queries_lens) if queries_lens else 0,
            average_query_length=total_query_len / len(queries_lens)
            if queries_lens
            else 0,
            max_query_length=max(queries_lens) if queries_lens else 0,
            unique_queries=unique_queries,
            num_query_images=num_query_images,
            min_query_image_width=min(q_img_widths) if q_img_widths else 0,
            average_query_image_width=sum(q_img_widths) / len(q_img_widths)
            if q_img_widths
            else 0,
            max_query_image_width=max(q_img_widths) if q_img_widths else 0,
            min_query_image_height=min(q_img_heights) if q_img_heights else 0,
            average_query_image_height=sum(q_img_heights) / len(q_img_heights)
            if q_img_heights
            else 0,
            max_query_image_height=max(q_img_heights) if q_img_heights else 0,
            min_relevant_docs_per_query=min(qrels_lengths),
            average_relevant_docs_per_query=qrels_per_doc,
            max_relevant_docs_per_query=max(qrels_lengths),
            unique_relevant_docs=unique_qrels,
        )


def process_language(relevant_docs, queries, corpus, lang=None):
    """We want to get three pieces of information:
    - the number of documents (and their char length) in the corpus
    - the number of queries (and their char length)
    - the average number of relevant documents per query
    """
    query_len, doc_len = calculate_length(queries, corpus)
    num_documents = len(corpus)
    num_queries = len(queries)

    # number of qrels that are not 0
    num_qrels_non_zero = sum(
        sum(1 for doc_id in docs if docs[doc_id] != 0)
        for docs in relevant_docs.values()
    )
    qrels_per_doc = num_qrels_non_zero / num_queries if num_queries else 0

    language_description = f" for language {lang}" if lang else ""
    logger.info(f"Average document character length{language_description} is {doc_len}")
    logger.info(f"Average query character length{language_description} is {query_len}")
    logger.info(f"Number of documents{language_description} is {num_documents}")
    logger.info(f"Number of queries{language_description} is {num_queries}")
    logger.info(
        f"Average number of relevant documents per query{language_description} is {qrels_per_doc}"
    )
    return {
        "average_document_length": doc_len,
        "average_query_length": query_len,
        "num_documents": num_documents,
        "num_queries": num_queries,
        "average_relevant_docs_per_query": qrels_per_doc,
    }


def calculate_length(queries, corpus):
    queries_lens = []
    doc_lens = []
    for query in queries:
        queries_lens.append(len(query))

    for doc in corpus:
        if isinstance(doc, Image.Image):
            doc_lens.append(1.0)  # for image append 1. Can perhaps be removed.

    doc_len = sum(doc_lens) / len(doc_lens) if doc_lens else 0
    query_len = sum(queries_lens) / len(queries_lens) if queries_lens else 0
    return query_len, doc_len


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


def process_docs(
    collection: dict[str, dict[str, dict[str, str] | str]], hf_subset: str, split: str
) -> dict[str, str]:
    """Collections can contain overlapping ids in different splits. Prepend split to avoid this"""
    return {
        f"{split}_{hf_subset}_{k}": v for k, v in collection[hf_subset][split].items()
    }


class MultiChoiceEvaluationMixin:
    """A mixin class to enable retrieval tasks to use multiple-choice evaluator;
    It is designed for tasks like r-Oxford and r-Pairs that
    require masking out different documents in the corpus for each query.

    example usage:
    class ROxfordHardI2IRetrieval(MultiChoiceEvaluationMixin, AbsTaskAny2AnyRetrieval):

    It is for overriding `def evaluate`, `def _evaluate_subset`
    and `def _calculate_metrics_from_split` of AbsTaskAny2AnyRetrieval.
    """

    def evaluate(
        self,
        model,
        split: str = "test",
        *,
        encode_kwargs: dict[str, Any] = None,
        **kwargs,
    ):
        # Use Any2AnyMultiChoiceEvaluator instead of Any2AnyRetrievalEvaluator
        evaluator = Any2AnyMultiChoiceEvaluator(
            retriever=model,
            task_name=self.metadata.name,
            encode_kwargs=encode_kwargs if encode_kwargs is not None else {},
            **kwargs,
        )

        scores = {}
        hf_subsets = list(self.hf_subsets) if self.is_multilingual else ["default"]

        for hf_subset in hf_subsets:
            logger.info(f"Subset: {hf_subset}")

            if hf_subset == "default":
                corpus, queries, relevant_docs = (
                    self.corpus[split],
                    self.queries[split],
                    self.relevant_docs[split],
                )
            else:
                corpus, queries, relevant_docs = (
                    self.corpus[hf_subset][split],
                    self.queries[hf_subset][split],
                    self.relevant_docs[hf_subset][split],
                )
            scores[hf_subset] = self._evaluate_subset(
                evaluator, corpus, queries, relevant_docs, hf_subset, **kwargs
            )
        return scores

    def _evaluate_subset(
        self, retriever, corpus, queries, relevant_docs, hf_subset: str, **kwargs
    ):
        start_time = time()
        results = retriever(corpus, queries, relevant_docs)
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

        ndcg, _map, recall, precision, cv_recall, naucs = retriever.evaluate(
            relevant_docs,
            results,
            retriever.k_values,
            ignore_identical_ids=self.ignore_identical_ids,
            skip_first_result=self.skip_first_result,
        )
        mrr, naucs_mrr = retriever.evaluate_custom(
            relevant_docs, results, retriever.k_values, "mrr"
        )
        scores = {
            **{f"ndcg_at_{k.split('@')[1]}": v for (k, v) in ndcg.items()},
            **{f"map_at_{k.split('@')[1]}": v for (k, v) in _map.items()},
            **{f"recall_at_{k.split('@')[1]}": v for (k, v) in recall.items()},
            **{f"cv_recall_at_{k.split('@')[1]}": v for (k, v) in cv_recall.items()},
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
            "accuracy": recall["Recall@1"],
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

    def _calculate_metrics_from_split(
        self, split: str, hf_subset: str | None = None, compute_overall: bool = False
    ) -> Any2AnyMutipleChoiceDescriptiveStatistics:
        if hf_subset:
            queries = self.queries[hf_subset][split]
            corpus = self.corpus[hf_subset][split]
            relevant_docs = self.relevant_docs[hf_subset][split]
        elif compute_overall:
            queries = {}
            corpus = {}
            relevant_docs = {}
            for hf_subset in self.metadata.eval_langs:
                queries.update(process_docs(self.queries, hf_subset, split))
                corpus.update(process_docs(self.corpus, hf_subset, split))
                relevant_docs.update(
                    process_relevant_docs(self.relevant_docs, hf_subset, split)
                )
        else:
            queries = self.queries[split]
            corpus = self.corpus[split]
            relevant_docs = self.relevant_docs[split]

        queries_lens, doc_lens = [], []
        num_query_images = 0
        num_document_images = 0

        q_modality = queries[0]["modality"]
        unique_queries = len(set(queries["text"])) if "text" in q_modality else 0

        for query in tqdm.tqdm(queries, desc="queries:"):
            if "text" in q_modality:
                text_query = query["text"]
                queries_lens.append(len(text_query))
            if "image" in q_modality:
                num_query_images += 1

        d_modality = corpus[0]["modality"]
        unique_documents = len(set(corpus["text"])) if "text" in d_modality else 0

        for doc in tqdm.tqdm(corpus, desc="docs:"):
            if "text" in d_modality:
                text_doc = doc["text"]
                doc_lens.append(len(text_doc))
            if "image" in d_modality:
                num_document_images += 1

        total_doc_len = sum(doc_lens)
        total_query_len = sum(queries_lens)
        num_documents = len(corpus)
        num_queries = len(queries)

        d_modality = corpus[0]["modality"]
        imgs = [doc["image"] for doc in corpus if "image" in d_modality]
        d_img_widths, d_img_heights = [], []
        for img in imgs:
            width, height = img.size
            d_img_widths.append(height)
            d_img_heights.append(width)

        q_modality = queries[0]["modality"]
        imgs = [query["image"] for query in queries if "image" in q_modality]
        q_img_widths, q_img_heights = [], []
        for img in imgs:
            width, height = img.size
            q_img_widths.append(height)
            q_img_heights.append(width)

        # create a list of number of relevant docs per query
        queries_set = set(queries["id"])
        qrels_lengths = [
            len([v for k, v in relevant_docs[qid].items() if v != 0])
            for qid in tqdm.tqdm(relevant_docs.keys(), desc="qrels:")
            if qid in queries_set
        ]
        num_qrels = sum(qrels_lengths)
        qrels_per_doc = num_qrels / len(relevant_docs) if num_queries else 0
        unique_qrels = len({doc for qid in relevant_docs for doc in relevant_docs[qid]})

        return Any2AnyMutipleChoiceDescriptiveStatistics(
            number_of_characters=total_query_len + total_doc_len,
            num_samples=num_documents + num_queries,
            num_queries=num_queries,
            num_documents=num_documents,
            min_document_length=min(doc_lens) if doc_lens else 0,
            average_document_length=total_doc_len / len(doc_lens) if doc_lens else 0,
            max_document_length=max(doc_lens) if doc_lens else 0,
            unique_documents=unique_documents,
            min_document_image_width=min(d_img_widths) if d_img_widths else 0,
            average_document_image_width=sum(d_img_widths) / len(d_img_widths)
            if d_img_widths
            else 0,
            max_document_image_width=max(d_img_widths) if d_img_widths else 0,
            min_document_image_height=min(d_img_heights) if d_img_heights else 0,
            average_document_image_height=sum(d_img_heights) / len(d_img_heights)
            if d_img_heights
            else 0,
            max_document_image_height=max(d_img_heights) if d_img_heights else 0,
            num_document_images=num_document_images,
            min_query_length=min(queries_lens) if queries_lens else 0,
            average_query_length=total_query_len / len(queries_lens)
            if queries_lens
            else 0,
            max_query_length=max(queries_lens) if queries_lens else 0,
            unique_queries=unique_queries,
            num_query_images=num_query_images,
            min_query_image_width=min(q_img_widths) if q_img_widths else 0,
            average_query_image_width=sum(q_img_widths) / len(q_img_widths)
            if q_img_widths
            else 0,
            max_query_image_width=max(q_img_widths) if q_img_widths else 0,
            min_query_image_height=min(q_img_heights) if q_img_heights else 0,
            average_query_image_height=sum(q_img_heights) / len(q_img_heights)
            if q_img_heights
            else 0,
            max_query_image_height=max(q_img_heights) if q_img_heights else 0,
            min_relevant_docs_per_query=min(qrels_lengths),
            average_relevant_docs_per_query=qrels_per_doc,
            max_relevant_docs_per_query=max(qrels_lengths),
            unique_relevant_docs=unique_qrels,
        )
