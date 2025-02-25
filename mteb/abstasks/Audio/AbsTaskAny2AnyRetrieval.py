from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from time import time
from typing import Any

import torch
import tqdm
from datasets import Features, Value, load_dataset

from ...evaluation.evaluators import Any2AnyRetrievalEvaluator
from ..AbsTask import AbsTask, ScoresDict

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
            self.hf_repo_qrels = hf_repo_qrels if hf_repo_qrels else hf_repo
        else:
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
            raise ValueError(f"File {fIn} must have extension {ext}")

    def load(
        self, split="test"
    ) -> tuple[
        dict[str, dict[str, str | torch.Tensor]],
        dict[str, dict[str, str | torch.Tensor]],
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
            logger.info(
                "Loaded %d Documents for %s split.", len(self.corpus), split.upper()
            )
            logger.info("Doc Example: %s", self.corpus[0])

        if not len(self.queries):
            logger.info("Loading Queries...")
            self._load_queries(split)

        self._load_qrels(split)
        qrels_dict = defaultdict(dict)

        def qrels_dict_init(row):
            qrels_dict[row["query-id"]][row["corpus-id"]] = int(row["score"])

        self.qrels.map(qrels_dict_init)
        self.qrels = qrels_dict
        self.queries = self.queries.filter(lambda x: x["id"] in self.qrels)
        logger.info("Loaded %d Queries for %s split.", len(self.queries), split.upper())
        logger.info("Query Example: %s", self.queries[0])
        return self.corpus, self.queries, self.qrels

    def load_corpus(self) -> dict[str, dict[str, str]]:
        if not self.hf_repo:
            self.check(fIn=self.corpus_file, ext="jsonl")
        if not len(self.corpus):
            logger.info("Loading Corpus...")
            self._load_corpus()
            logger.info("Loaded %d Documents.", len(self.corpus))
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
        qrels_ds = qrels_ds.select_columns(["query-id", "corpus-id", "score"]).cast(
            features
        )
        self.qrels = qrels_ds


class AbsTaskAny2AnyRetrieval(AbsTask):
    """Abstract class for audio-text retrieval experiments.

    Child-classes must implement:
      - self.corpus: dict[str, dict[str, str]]
          e.g. {"test": {"doc1": {"_id": "d1", "title": "title", "audio": "audio_path or data"}}}
      - self.queries: dict[str, dict[str, str]]
          e.g. {"test": {"q1": "query text"}} or conversational queries
      - self.relevant_docs: dict[str, dict[str, dict[str, int]]]
          e.g. {"test": {"q1": {"doc1": 1}}}
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
        retriever = Any2AnyRetrievalEvaluator(
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
        results = retriever(corpus, queries)
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
            predictions_path = (
                output_folder / f"{self.metadata.name}_{hf_subset}_predictions.json"
            )
            with open(predictions_path, "w") as f:
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
        }
        self._add_main_score(scores)

        if export_errors:
            errors = {}
            top_k = kwargs.get("top_k", 1)
            if not save_predictions and top_k == 1:
                for qid in results.keys():
                    sorted_docs = sorted(
                        results[qid].items(), key=lambda x: x[1], reverse=True
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
            errors_path = (
                output_folder / f"{self.metadata.name}_{hf_subset}_errors.json"
            )
            with open(errors_path, "w") as f:
                json.dump(errors, f)

        return scores

    def _add_main_score(self, scores: ScoresDict) -> None:
        scores["main_score"] = scores[self.metadata.main_score]

    def _calculate_metrics_from_split(
        self, split: str, hf_subset: str | None = None, compute_overall: bool = False
    ):
        pass

    def calculate_metadata_metrics(self) -> None:
        self.load_data()
        all_details = {}
        pbar_split = tqdm.tqdm(
            self.metadata_dict["eval_splits"], desc="Processing Splits..."
        )
        for split in pbar_split:
            pbar_split.set_postfix_str(f"Split: {split}")
            logger.info(f"Processing metadata for split {split}")
            all_details[split] = {}
            if self.is_multilingual:
                pbar_lang = tqdm.tqdm(
                    self.relevant_docs.keys(), desc="Processing Languages..."
                )
                for lang in pbar_lang:
                    pbar_lang.set_postfix_str(f"Language: {lang}")
                    logger.info(f"Processing metadata for language {lang}")
                    split_details = process_language(
                        self.relevant_docs[lang][split],
                        self.queries[lang][split],
                        self.corpus[lang][split],
                        lang,
                    )
                    all_details[split][lang] = split_details
            else:
                split_details = process_language(
                    self.relevant_docs[split], self.queries[split], self.corpus[split]
                )
                all_details[split] = split_details
        return all_details


def process_language(relevant_docs, queries, corpus, lang=None):
    query_len, doc_len = calculate_length(queries, corpus)
    num_documents = len(corpus)
    num_queries = len(queries)
    num_qrels_non_zero = sum(
        sum(1 for doc_id in docs if docs[doc_id] != 0)
        for docs in relevant_docs.values()
    )
    qrels_per_doc = num_qrels_non_zero / num_queries if num_queries else 0
    language_description = f" for language {lang}" if lang else ""
    logger.info(f"Average document length{language_description} is {doc_len}")
    logger.info(f"Average query length{language_description} is {query_len}")
    logger.info(f"Number of documents{language_description} is {num_documents}")
    logger.info(f"Number of queries{language_description} is {num_queries}")
    logger.info(
        f"Average relevant docs per query{language_description} is {qrels_per_doc}"
    )
    return {
        "average_document_length": doc_len,
        "average_query_length": query_len,
        "num_documents": num_documents,
        "num_queries": num_queries,
        "average_relevant_docs_per_query": qrels_per_doc,
    }


def calculate_length(queries, corpus):
    queries_lens = [len(query) for query in queries.values()]
    doc_lens = []
    for doc in corpus.values():
        # For audio, assuming doc is a tensor; using 1.0 as a placeholder length.
        if isinstance(doc, torch.Tensor):
            doc_lens.append(1.0)
    doc_len = sum(doc_lens) / len(doc_lens) if doc_lens else 0
    query_len = sum(queries_lens) / len(queries_lens) if queries_lens else 0
    return query_len, doc_len
