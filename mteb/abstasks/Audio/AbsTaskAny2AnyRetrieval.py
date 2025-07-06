from __future__ import annotations

import json
import logging
import os
import warnings
from collections import defaultdict
from pathlib import Path
from time import time
from typing import Any

import datasets
import numpy as np
from datasets import Dataset, Features, Value, load_dataset

from mteb.abstasks.TaskMetadata import HFSubset

from ...evaluation.evaluators.Audio.Any2AnyRetrievalEvaluator import (
    Any2AnyRetrievalEvaluator,
)
from ...load_results.task_results import ScoresDict
from ..AbsTask import AbsTask
from ..TaskMetadata import DescriptiveStatistics

logger = logging.getLogger(__name__)


# Adapted from https://github.com/beir-cellar/beir/blob/f062f038c4bfd19a8ca942a9910b1e0d218759d4/beir/datasets/data_loader_hf.py#L10
class HFDataLoader:
    def __init__(
        self,
        hf_repo: str | None = None,
        hf_repo_qrels: str | None = None,
        data_folder: str | None = None,
        prefix: str | None = None,
        corpus_file: str = "corpus.jsonl",
        query_file: str = "queries.jsonl",
        qrels_folder: str = None,
        qrels_file: str = None,
        streaming: bool = False,
        keep_in_memory: bool = False,
        trust_remote_code: bool = False,
        id_column_name: str = "_id",
        audio_column_name: str = "audio",
        text_column_name: str = "text",
    ):
        self.corpus = {}
        self.queries = {}
        self.qrels = {}
        self.qrels_file = None
        self.hf_repo = hf_repo
        self.audio_column_name = audio_column_name
        self.id_column_name = id_column_name
        self.text_column_name = text_column_name
        self.trust_remote_code = trust_remote_code
        if hf_repo:
            # By default fetch qrels from same repo not a second repo with "-qrels" like in original
            self.hf_repo_qrels = hf_repo_qrels if hf_repo_qrels else hf_repo
        else:
            warnings.warn(
                "Loading from local files will be removed in v2.0.0.",
                DeprecationWarning,
            )
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
                os.path.join(data_folder, qrels_folder)
                if data_folder and qrels_folder
                else None
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
    ) -> tuple[dict[str, dict[str, str]], dict[str, str], dict[str, dict[str, int]]]:
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
            self._load_queries()

        if self.qrels_file is not None:
            self._load_qrels(split)
        else:
            self._generate_qrels(split)
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

    def _generate_qrels(self, split: str):
        ds = load_dataset(
            self.hf_repo,
            "default",
            keep_in_memory=self.keep_in_memory,
            streaming=self.streaming,
            trust_remote_code=self.trust_remote_code,
        )[split]

        qrels_dict = {
            "query-id": ds[self.id_column_name],
            "corpus-id": ds[self.id_column_name],
            "score": [1] * len(ds),
        }
        features = Features(
            {
                "query-id": Value("string"),
                "corpus-id": Value("string"),
                "score": Value("int32"),
            }
        )
        self.qrels = Dataset.from_dict(qrels_dict, features=features)

    def _load_corpus(self):
        if self.hf_repo:
            datasets.logging.set_verbosity_debug()
            # print(self.hf_repo)
            corpus_ds = load_dataset(
                self.hf_repo,
                "default",
                keep_in_memory=self.keep_in_memory,
                streaming=self.streaming,
                trust_remote_code=self.trust_remote_code,
            )
        else:
            corpus_ds = load_dataset(
                "json",
                data_files=self.corpus_file,
                streaming=self.streaming,
                keep_in_memory=self.keep_in_memory,
            )
        corpus_ds = next(iter(corpus_ds.values()))  # get first split
        # print(corpus_ds)
        corpus_ds = corpus_ds.cast_column(self.id_column_name, Value(dtype="string"))
        corpus_ds = corpus_ds.rename_column(self.id_column_name, "id")
        corpus_ds = corpus_ds.remove_columns(
            [
                col
                for col in corpus_ds.column_names
                if col not in ["id", self.text_column_name, "title"]
            ]
        )
        self.corpus = corpus_ds

    def _load_queries(self):
        if self.hf_repo:
            queries_ds = load_dataset(
                self.hf_repo,
                "default",
                keep_in_memory=self.keep_in_memory,
                streaming=self.streaming,
                trust_remote_code=self.trust_remote_code,
            )
        else:
            queries_ds = load_dataset(
                "json",
                data_files=self.query_file,
                streaming=self.streaming,
                keep_in_memory=self.keep_in_memory,
            )
        queries_ds = next(iter(queries_ds.values()))
        queries_ds = queries_ds.cast_column(self.id_column_name, Value("string"))
        queries_ds = queries_ds.rename_column(self.id_column_name, "id")
        queries_ds = queries_ds.remove_columns(
            [
                col
                for col in queries_ds.column_names
                if col not in ["id", self.audio_column_name]
            ]
        )
        self.queries = queries_ds

    def _load_qrels(self, split):
        if self.hf_repo:
            qrels_ds = load_dataset(
                self.hf_repo_qrels,
                keep_in_memory=self.keep_in_memory,
                streaming=self.streaming,
                trust_remote_code=self.trust_remote_code,
            )[split]
        else:
            qrels_ds = load_dataset(
                "csv",
                data_files=self.qrels_file,
                delimiter="\t",
                keep_in_memory=self.keep_in_memory,
            )
        features = Features(
            {
                "query-id": Value("string"),
                "corpus-id": Value("string"),
                "score": Value("float"),
            }
        )
        print(qrels_ds)
        qrels_ds = qrels_ds.cast(features)
        self.qrels = qrels_ds


class Any2AnyRetrievalDescriptiveStatistics(DescriptiveStatistics):
    """Descriptive statistics for Retrieval

    Attributes:
        num_samples: Number of queries and documents
        num_queries: number of queries in the dataset
        num_documents: Number of documents
        number_of_characters: Total number of symbols in the dataset

        min_document_length: Minimum length of documents
        average_document_length: Average length of documents
        max_document_length: Maximum length of documents
        unique_documents: Number of unique documents

        min_query_length: Minimum length of queries
        average_query_length: Average length of queries
        max_query_length: Maximum length of queries
        unique_queries: Number of unique queries

        min_relevant_docs_per_query: Minimum number of relevant documents per query
        average_relevant_docs_per_query: Average number of relevant documents per query
        max_relevant_docs_per_query: Maximum number of relevant documents per query
        unique_relevant_docs: Number of unique relevant documents

        num_document_audio: Number of audio documents
        num_query_audio: Number of audio queries
    """

    num_samples: int
    num_queries: int
    num_documents: int
    number_of_characters: int

    min_document_length: int
    average_document_length: float
    max_document_length: int
    unique_documents: int

    min_query_length: int
    average_query_length: float
    max_query_length: int
    unique_queries: int

    min_relevant_docs_per_query: int
    average_relevant_docs_per_query: float
    max_relevant_docs_per_query: int
    unique_relevant_docs: int

    num_document_audio: int
    num_query_audio: int


class AbsTaskAny2AnyRetrieval(AbsTask):
    """Abstract class for audio to text retrieval experiments.

    Child-classes must implement the following properties:

    self.corpus: dict[str, dict[str, str]]
        Semantically, it should contain dict[split_name, dict[sample_id, dict[str, str]]]
        E.g. {"test": {"document_one": {"_id": "d1", "title": "title", "text": "text"}}}

    self.queries: dict[str, dict[str, Union[np.array, list[np.array]]]]
        Semantically, it should contain dict[split_name, dict[sample_id, np.array]] or dict[split_name, dict[sample_id, list[np.array]]] for conversations
        E.g. {"test": {"q1": "query"}}
        or {"test": {"q1": ["turn1", "turn2", "turn3"]}}

    self.relevant_docs: dict[str, dict[str, dict[str, int]]]
        Semantically, it should contain dict[split_name, dict[sample_id, dict[doc_id, score]]]
        E.g.: {"test": {"q1": {"document_one": 1}}}
    """

    ignore_identical_ids: bool = False
    abstask_prompt = "Retrieve text based on user query."

    # column names for different modalities (configurable)
    audio_column_name: str = "audio"
    text_column_name: str = "caption"
    id_column_name: str = "audiocap_id"

    # modality detection settings
    modality_column_name: str = (
        "modality"  # column that specifies modality if available
    )
    default_query_modality: str = "audio"
    default_corpus_modality: str = "text"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _detect_modalities(self, dataset_split) -> tuple[list[str], list[str]]:
        """Detect modalities for queries and corpus from dataset structure.

        Returns:
            tuple: (query_modalities, corpus_modalities) - lists of modality strings
        """
        # check if dataset has explicit modality information
        if self.modality_column_name in dataset_split.column_names:
            # use explicit modality info from the dataset
            first_item = dataset_split[0]
            modality_info = first_item[self.modality_column_name]

            # handle different modality info formats
            if isinstance(modality_info, dict):
                # format: {"query": ["audio"], "corpus": ["text"]}
                query_modalities = modality_info.get(
                    "query", [self.default_query_modality]
                )
                corpus_modalities = modality_info.get(
                    "corpus", [self.default_corpus_modality]
                )
            else:
                # format: single modality string or list for both
                if isinstance(modality_info, str):
                    modality_info = [modality_info]
                query_modalities = modality_info
                corpus_modalities = modality_info

            return query_modalities, corpus_modalities
        else:
            return [self.default_query_modality], [self.default_corpus_modality]

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        # first load the dataset and apply transforms like base class
        import datasets

        self.dataset = datasets.load_dataset(**self.metadata_dict["dataset"])
        self.dataset_transform()

        # now extract corpus, queries, and qrels from the transformed dataset
        self.corpus, self.queries, self.relevant_docs = {}, {}, {}
        # dataset_path = self.metadata_dict["dataset"]["path"]

        for split_config in kwargs.get(
            "eval_splits", self.metadata_dict["eval_splits"]
        ):
            # handle split configs like "test[:%10]" -> "test"
            split = split_config.split("[")[0]

            if split not in self.dataset:
                continue

            # extract data directly from transformed dataset
            dataset_split = self.dataset[split]

            query_modalities, corpus_modalities = self._detect_modalities(dataset_split)

            # build corpus and queries from transformed dataset
            corpus = {}
            queries = {}
            qrels = {}

            for i, item in enumerate(dataset_split):
                item_id = item[self.id_column_name]

                # create corpus entry based on detected modality
                if "text" in corpus_modalities:
                    title = item.get("title", "")
                    text = item[self.text_column_name]
                    corpus[item_id] = (title + " " + text).strip()
                elif "audio" in corpus_modalities:
                    audio_data = item[self.audio_column_name]
                    corpus[item_id] = audio_data

                # create query entry based on detected modality
                if "audio" in query_modalities:
                    audio_data = item[self.audio_column_name]
                    queries[item_id] = audio_data
                elif "text" in query_modalities:
                    title = item.get("title", "")
                    text = item[self.text_column_name]
                    queries[item_id] = (title + " " + text).strip()

                # create qrels (default 1:1 mapping for now)
                qrels[item_id] = {item_id: 1}

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
        subsets_to_run: list[HFSubset] | None = None,
        *,
        encode_kwargs: dict[str, Any] = {},
        **kwargs,
    ) -> dict[HFSubset, ScoresDict]:
        retriever = Any2AnyRetrievalEvaluator(
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

            # handle split configs like "test[:%10]" -> "test"
            actual_split = split.split("[")[0]

            if hf_subset == "default":
                corpus, queries, relevant_docs = (
                    self.corpus[actual_split],
                    self.queries[actual_split],
                    self.relevant_docs[actual_split],
                )
            else:
                corpus, queries, relevant_docs = (
                    self.corpus[hf_subset][actual_split],
                    self.queries[hf_subset][actual_split],
                    self.relevant_docs[hf_subset][actual_split],
                )
            scores[hf_subset] = self._evaluate_subset(
                retriever, corpus, queries, relevant_docs, hf_subset, **kwargs
            )
        return scores

    def _evaluate_subset(
        self, retriever, corpus, queries, relevant_docs, hf_subset: str, **kwargs
    ) -> ScoresDict:
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
            qrels_save_path = (
                output_folder / f"{self.metadata.name}_{hf_subset}_predictions.json"
            )

            with open(qrels_save_path, "w") as f:
                json.dump(results, f)

        ndcg, _map, recall, precision, naucs = retriever.evaluate(
            relevant_docs,
            results,
            retriever.k_values,
            ignore_identical_ids=self.ignore_identical_ids,
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
    ) -> Any2AnyRetrievalDescriptiveStatistics:
        # handle split configs like "test[:%10]" -> "test"
        actual_split = split.split("[")[0]

        if hf_subset:
            queries = self.queries[hf_subset][actual_split]
            corpus = self.corpus[hf_subset][actual_split]
            relevant_docs = self.relevant_docs[hf_subset][actual_split]
        elif compute_overall:
            queries = {}
            corpus = {}
            relevant_docs = {}
            for hf_subset in self.metadata.eval_langs:
                queries.update(process_docs(self.queries, hf_subset, actual_split))
                corpus.update(process_docs(self.corpus, hf_subset, actual_split))
                relevant_docs.update(
                    process_relevant_docs(self.relevant_docs, hf_subset, actual_split)
                )
        else:
            queries = self.queries[actual_split]
            corpus = self.corpus[actual_split]
            relevant_docs = self.relevant_docs[actual_split]

        dataset_split = self.dataset[actual_split]
        query_modalities, corpus_modalities = self._detect_modalities(dataset_split)

        query_lens, doc_lens = [], []
        num_query_audio = 0
        num_document_audio = 0
        unique_queries = 0
        unique_documents = 0

        # calculate query stats based on modalities
        if "text" in query_modalities:
            # text queries - calculate lengths
            for query in queries.values():
                if isinstance(query, str):
                    query_lens.append(len(query))
            unique_queries = len(set(queries.values()))
        elif "audio" in query_modalities:
            # audio queries - count them
            num_query_audio = len(queries)
            # for audio, we can't easily compute text lengths, so use 0
            query_lens = [0] * num_query_audio

        if "text" in corpus_modalities:
            for doc in corpus.values():
                if isinstance(doc, str):
                    doc_lens.append(len(doc))
            unique_documents = len(set(corpus.values()))
        elif "audio" in corpus_modalities:
            num_document_audio = len(corpus)
            doc_lens = [0] * num_document_audio
        num_documents = len(corpus)
        num_queries = len(queries)

        # create a list of number of relevant docs per query
        qrels_lengths = [
            len(relevant_docs[qid]) for qid in relevant_docs if qid in queries
        ]
        num_qrels = sum(qrels_lengths)
        qrels_per_doc = num_qrels / len(relevant_docs) if num_queries else 0
        unique_qrels = len({doc for qid in relevant_docs for doc in relevant_docs[qid]})

        return {
            "number_of_characters": sum(query_lens) + sum(doc_lens),
            "num_samples": num_documents + num_queries,
            "num_queries": num_queries,
            "num_documents": num_documents,
            "min_document_length": min(doc_lens) if doc_lens else 0,
            "average_document_length": sum(doc_lens) / num_documents if doc_lens else 0,
            "max_document_length": max(doc_lens) if doc_lens else 0,
            "unique_documents": unique_documents,
            "num_document_audio": num_document_audio,
            "min_query_length": min(query_lens) if query_lens else 0,
            "average_query_length": sum(query_lens) / num_queries if query_lens else 0,
            "max_query_length": max(query_lens) if query_lens else 0,
            "unique_queries": unique_queries,
            "num_query_audio": num_query_audio,
            "min_relevant_docs_per_query": min(qrels_lengths) if qrels_lengths else 0,
            "average_relevant_docs_per_query": qrels_per_doc,
            "max_relevant_docs_per_query": max(qrels_lengths) if qrels_lengths else 0,
            "unique_relevant_docs": unique_qrels,
        }


def calculate_length(
    queries: dict[str, np.array], corpus: dict[str, str]
) -> tuple[list[int], list[int]]:
    queries_lens = []
    doc_lens = []
    for query in queries.values():
        if isinstance(query, str):
            # text query
            queries_lens.append(len(query))
        elif isinstance(query, np.ndarray):
            # audio query - use a default length of 1 for now
            queries_lens.append(1)
        elif isinstance(query, list) and len(query) > 0:
            # conversation-style query
            if isinstance(query[0], str):
                queries_lens.extend([len(turn) for turn in query])
            else:
                # audio conversation - use default length
                queries_lens.extend([1 for _ in query])
        else:
            # unknown type, use default length
            queries_lens.append(1)

    for doc in corpus.values():
        if isinstance(doc, str):
            # text document
            doc_lens.append(len(doc))
        elif isinstance(doc, np.ndarray):
            # audio document - use a default length of 1 for now
            doc_lens.append(1)
        else:
            # unknown type, use default length
            doc_lens.append(1)

    return queries_lens, doc_lens


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
