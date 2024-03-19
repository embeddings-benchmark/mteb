import json
import logging
import os
from collections import defaultdict
from time import time
from typing import Dict, Tuple

from datasets import Features, Value, load_dataset

from ..evaluation.evaluators import RetrievalEvaluator
from .AbsTask import AbsTask

logger = logging.getLogger(__name__)


# Adapted from https://github.com/beir-cellar/beir/blob/f062f038c4bfd19a8ca942a9910b1e0d218759d4/beir/datasets/data_loader_hf.py#L10
class HFDataLoader:
    def __init__(
        self,
        hf_repo: str = None,
        hf_repo_qrels: str = None,
        data_folder: str = None,
        prefix: str = None,
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
            raise ValueError(
                "File {} not present! Please provide accurate file.".format(fIn)
            )

        if not fIn.endswith(ext):
            raise ValueError(
                "File {} must be present with extension {}".format(fIn, ext)
            )

    def load(
        self, split="test"
    ) -> Tuple[Dict[str, Dict[str, str]], Dict[str, str], Dict[str, Dict[str, int]]]:
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

    def load_corpus(self) -> Dict[str, Dict[str, str]]:
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
            )
        else:
            corpus_ds = load_dataset(
                "json",
                data_files=self.corpus_file,
                streaming=self.streaming,
                keep_in_memory=self.keep_in_memory,
            )
        corpus_ds = next(iter(corpus_ds.values()))  # get first split
        corpus_ds = corpus_ds.cast_column("_id", Value("string"))
        corpus_ds = corpus_ds.rename_column("_id", "id")
        corpus_ds = corpus_ds.remove_columns(
            [
                col
                for col in corpus_ds.column_names
                if col not in ["id", "text", "title"]
            ]
        )
        self.corpus = corpus_ds

    def _load_queries(self):
        if self.hf_repo:
            queries_ds = load_dataset(
                self.hf_repo,
                "queries",
                keep_in_memory=self.keep_in_memory,
                streaming=self.streaming,
            )
        else:
            queries_ds = load_dataset(
                "json",
                data_files=self.query_file,
                streaming=self.streaming,
                keep_in_memory=self.keep_in_memory,
            )
        queries_ds = next(iter(queries_ds.values()))  # get first split
        queries_ds = queries_ds.cast_column("_id", Value("string"))
        queries_ds = queries_ds.rename_column("_id", "id")
        queries_ds = queries_ds.remove_columns(
            [col for col in queries_ds.column_names if col not in ["id", "text"]]
        )
        self.queries = queries_ds

    def _load_qrels(self, split):
        if self.hf_repo:
            qrels_ds = load_dataset(
                self.hf_repo_qrels,
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
        features = Features(
            {
                "query-id": Value("string"),
                "corpus-id": Value("string"),
                "score": Value("float"),
            }
        )
        qrels_ds = qrels_ds.cast(features)
        self.qrels = qrels_ds


class AbsTaskRetrieval(AbsTask):
    """
    Abstract class for re-ranking experiments.

    Child-classes must implement the following properties:

    self.corpus: dict[str, dict[str, str]]
        Semantically, it should contain dict[split_name, dict[sample_id, dict[str, str]]]
        E.g. {"test": {"document_one": {"_id": "d1", "title": "title", "text": "text"}}}

    self.queries: dict[str, dict[str, str]]
        Semantically, it should contain dict[split_name, dict[sample_id, str]]
        E.g. {"test": {"q1": "query"}}

    self.relevant_docs: dict[str, dict[str, dict[str, int]]]
        Semantically, it should contain dict[split_name, dict[sample_id, dict[doc_id, score]]]
        E.g.: {"test": {"q1": {"document_one": 1}}}
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load_data(self, **kwargs):
        if self.data_loaded:
            return
        self.corpus, self.queries, self.relevant_docs = {}, {}, {}
        hf_repo_qrels = (
            self.metadata_dict["hf_hub_name"] + "-qrels"
            if "clarin-knext" in self.metadata_dict["hf_hub_name"]
            else None
        )
        for split in kwargs.get("eval_splits", self.metadata_dict["eval_splits"]):
            corpus, queries, qrels = HFDataLoader(
                hf_repo=self.metadata_dict["hf_hub_name"],
                hf_repo_qrels=hf_repo_qrels,
                streaming=False,
                keep_in_memory=False,
            ).load(split=split)
            # Conversion from DataSet
            queries = {query["id"]: query["text"] for query in queries}
            corpus = {
                doc["id"]: {"title": doc["title"], "text": doc["text"]}
                for doc in corpus
            }
            self.corpus[split], self.queries[split], self.relevant_docs[split] = (
                corpus,
                queries,
                qrels,
            )

        self.data_loaded = True

    def evaluate(self, model, split="test", **kwargs):
        retriever = RetrievalEvaluator(model, **kwargs)

        scores = {}
        if self.is_multilingual:
            for lang in self.langs:
                logger.info(f"Language: {lang}")
                corpus, queries, relevant_docs = (
                    self.corpus[lang][split],
                    self.queries[lang][split],
                    self.relevant_docs[lang][split],
                )
                scores[lang] = self._evaluate_monolingual(
                    retriever, corpus, queries, relevant_docs, lang, **kwargs
                )
        else:
            corpus, queries, relevant_docs = (
                self.corpus[split],
                self.queries[split],
                self.relevant_docs[split],
            )
            scores = self._evaluate_monolingual(
                retriever, corpus, queries, relevant_docs, None, **kwargs
            )
        return scores

    def _evaluate_monolingual(
        self, retriever, corpus, queries, relevant_docs, lang=None, **kwargs
    ):
        start_time = time()
        results = retriever(corpus, queries)
        end_time = time()
        logger.info(
            "Time taken to retrieve: {:.2f} seconds".format(end_time - start_time)
        )

        if kwargs.get("save_qrels", False):
            output_folder = kwargs.get("output_folder", "results")
            if not os.path.isdir(output_folder):
                os.makedirs(output_folder)
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
            if lang is None:
                qrels_save_path = (
                    f"{output_folder}/{self.metadata_dict['name']}_qrels.json"
                )
            else:
                qrels_save_path = (
                    f"{output_folder}/{self.metadata_dict['name']}_{lang}_qrels.json"
                )

            with open(qrels_save_path, "w") as f:
                json.dump(results, f)

        ndcg, _map, recall, precision = retriever.evaluate(
            relevant_docs,
            results,
            retriever.k_values,
            ignore_identical_ids=kwargs.get("ignore_identical_ids", True),
        )
        mrr = retriever.evaluate_custom(
            relevant_docs, results, retriever.k_values, "mrr"
        )
        scores = {
            **{f"ndcg_at_{k.split('@')[1]}": v for (k, v) in ndcg.items()},
            **{f"map_at_{k.split('@')[1]}": v for (k, v) in _map.items()},
            **{f"recall_at_{k.split('@')[1]}": v for (k, v) in recall.items()},
            **{f"precision_at_{k.split('@')[1]}": v for (k, v) in precision.items()},
            **{f"mrr_at_{k.split('@')[1]}": v for (k, v) in mrr.items()},
        }
        return scores
