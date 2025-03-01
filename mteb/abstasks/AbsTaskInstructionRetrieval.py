from __future__ import annotations

import json
import logging
import os
import warnings
from collections import defaultdict
from time import time
from typing import Any

import tqdm
from datasets import Dataset, Features, Value, load_dataset

from mteb.encoder_interface import Encoder

from ..evaluation.evaluators import utils
from ..evaluation.evaluators.InstructionRetrievalEvaluator import (
    InstructionRetrievalEvaluator,
)
from .AbsTask import AbsTask
from .AbsTaskRetrieval import HFDataLoader
from .TaskMetadata import DescriptiveStatistics

logger = logging.getLogger(__name__)


# Adapted from https://github.com/beir-cellar/beir/blob/f062f038c4bfd19a8ca942a9910b1e0d218759d4/beir/datasets/data_loader_hf.py#L10
class HFDataLoaderInstructions(HFDataLoader):
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
        trust_remote_code: bool = False,
    ):
        self.corpus = {}
        self.queries = {}
        self.qrels = {}
        self.og_instructions = {}
        self.changed_instructions = {}
        self.top_ranked = {}
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
        self.trust_remote_code = trust_remote_code

    def load(
        self, split="test"
    ) -> tuple[
        Dataset,
        Dataset,
        dict[str, dict[str, int]],
        dict[str, dict[str, int]],
        Dataset,
    ]:
        if not self.hf_repo:
            self.og_qrels_file = os.path.join(self.qrels_folder + "_og", split + ".tsv")
            self.changed_qrels_file = os.path.join(
                self.qrels_folder + "_changed", split + ".tsv"
            )
            self.check(fIn=self.corpus_file, ext="jsonl")
            self.check(fIn=self.query_file, ext="jsonl")
            self.check(fIn=self.og_qrels_file, ext="tsv")
            self.check(fIn=self.changed_qrels_file, ext="tsv")

        if not len(self.corpus):
            logger.info("Loading Corpus...")
            self._load_corpus()
            logger.info("Loaded %d %s Documents.", len(self.corpus), split.upper())
            logger.info("Doc Example: %s", self.corpus[0])

        if not len(self.queries):
            logger.info("Loading Queries...")
            self._load_queries()

        self._load_qrels(split, changed=False)
        self._load_qrels(split, changed=True)
        # filter queries with no qrels
        og_qrels_dict = defaultdict(dict)
        changed_qrels_dict = defaultdict(dict)

        def qrels_dict_init(row):
            og_qrels_dict[row["query-id"]][row["corpus-id"]] = int(row["score"])

        def qrels_changed_dict_init(row):
            changed_qrels_dict[row["query-id"]][row["corpus-id"]] = int(row["score"])

        self.changed_qrels.map(qrels_dict_init)
        self.og_qrels.map(qrels_changed_dict_init)
        self.og_qrels = og_qrels_dict
        self.changed_qrels = changed_qrels_dict
        self.queries = self.queries.filter(lambda x: x["id"] in self.og_qrels)
        logger.info("Loaded %d %s Queries.", len(self.queries), split.upper())
        logger.info("Query Example: %s", self.queries[0])

        # load top_ranked
        self.load_top_ranked()

        return (
            self.corpus,
            self.queries,
            self.og_qrels,
            self.changed_qrels,
            self.top_ranked,
        )

    def load_top_ranked(self) -> None:
        if self.hf_repo:
            top_ranked_ds = load_dataset(
                self.hf_repo,
                "top_ranked",
                keep_in_memory=self.keep_in_memory,
                streaming=self.streaming,
            )
        else:
            top_ranked_ds = load_dataset(
                "json",
                data_files=self.top_ranked_file,
                streaming=self.streaming,
                keep_in_memory=self.keep_in_memory,
            )
        top_ranked_ds = next(iter(top_ranked_ds.values()))  # get first split
        top_ranked_ds = top_ranked_ds.cast_column("qid", Value("string"))
        top_ranked_ds = top_ranked_ds.cast_column("pid", Value("string"))
        top_ranked_ds = top_ranked_ds.remove_columns(
            [col for col in top_ranked_ds.column_names if col not in ["qid", "pid"]]
        )
        self.top_ranked = top_ranked_ds

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
            [
                col
                for col in queries_ds.column_names
                if col
                not in [
                    "id",
                    "text",
                    "instruction_og",
                    "instruction_changed",
                    "keywords",
                    "short_query",
                ]
            ]
        )
        self.queries = queries_ds

    def _load_qrels(self, split, changed=False):
        if self.hf_repo:
            qrels_ds = load_dataset(
                self.hf_repo_qrels,
                "qrels_og" if not changed else "qrels_changed",
                keep_in_memory=self.keep_in_memory,
                streaming=self.streaming,
            )[split]
        else:
            qrels_file = self.og_qrels_file if not changed else self.changed_qrels_file
            qrels_ds = load_dataset(
                "csv",
                data_files=qrels_file,
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

        if changed:
            self.changed_qrels = qrels_ds
        else:
            self.og_qrels = qrels_ds


class InstructionRetrievalDescriptiveStatistics(DescriptiveStatistics):
    """Descriptive statistics for Instruction Retrieval tasks

    Attributes:
        num_samples: Number of samples
        num_queries: Number of queries
        num_docs: Number of documents
        number_of_characters: Total number of symbols in the dataset

        min_document_length: Minimum length of documents
        average_document_length: Average length of documents
        max_document_length: Maximum length of documents
        unique_docs: Number of unique documents

        min_query_length: Minimum length of queries
        average_query_length: Average length of queries
        max_query_length: Maximum length of queries
        unique_queries: Number of unique queries

        min_instruction_length: Minimum length of instructions
        average_instruction_length: Average length of instructions
        max_instruction_length: Maximum length of instructions
        unique_instructions: Number of unique instructions

        min_changed_instruction_length: Minimum length of changed instructions
        average_changed_instruction_length: Average length of changed instructions
        max_changed_instruction_length: Maximum length of changed instructions
        unique_changed_instructions: Number of unique changed instructions

        min_average_relevant_docs_per_query: Minimum number of relevant docs per query
        average_relevant_docs_per_query: Average number of relevant docs per query
        max_average_relevant_docs_per_query: Maximum number of relevant docs per query

        min_average_top_ranked_per_query: Minimum number of top ranked docs per query
        average_top_ranked_per_query: Average number of top ranked docs per query
        max_average_top_ranked_per_query: Maximum number of top ranked docs per query
    """

    num_samples: int
    num_queries: int
    num_docs: int
    number_of_characters: int

    min_document_length: int
    average_document_length: float
    max_document_length: int
    unique_docs: int

    min_query_length: int
    average_query_length: float
    max_query_length: int
    unique_queries: int

    min_instruction_length: int
    average_instruction_length: float
    max_instruction_length: int
    unique_instructions: int

    min_changed_instruction_length: int
    average_changed_instruction_length: float
    max_changed_instruction_length: int
    unique_changed_instructions: int

    min_average_relevant_docs_per_query: float
    average_relevant_docs_per_query: float
    max_average_relevant_docs_per_query: float

    min_average_top_ranked_per_query: float
    average_top_ranked_per_query: float
    max_average_top_ranked_per_query: float


class AbsTaskInstructionRetrieval(AbsTask):
    """Abstract class for retrieval tasks that use instructions. An example from Core17 would be
        query: What is the ongoing status of The Three Gorges Project?
        instruction: A relevant document will provide the projected or actual date of completion of the project, its estimated or actual total cost, or the estimated or ongoing electrical output of the finished project. Discussions of the social, political, or ecological impact of the project are not relevant.

    Child-classes must implement the following properties:
    self.corpus = dict[corpus_id, dict[str, str]] #id => dict with document datas like title and text
    self.queries = dict[query_id, str] #id => query
    self.relevant_docs = dict[query_id, dict[corpus_id, int]]
    self.og_instructions = dict[str, str] query => original instruction
    self.changed_instructions = dict[str, str] query => changed instruction
    self.top_ranked = dict[query_id, list[corpus_id]] #id => list of top ranked document ids

    See https://arxiv.org/abs/2403.15246 for more details
    """

    abstask_prompt = "Retrieve text based on user query."

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.do_length_ablation = kwargs.get("do_length_ablation", False)
        if self.do_length_ablation:
            logger.info("Running length ablation also...")
        warnings.warn(
            "`AbsTaskInstructionRetrieval` will be merged with Retrieval in v2.0.0.",
            DeprecationWarning,
        )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return
        self.corpus, self.queries, self.og_relevant_docs, self.changed_relevant_docs = (
            {},
            {},
            {},
            {},
        )
        self.og_instructions, self.changed_instructions = {}, {}
        self.top_ranked = {}
        if self.do_length_ablation:
            self.keywords, self.short_instructions = {}, {}

        dataset_path = self.metadata_dict["dataset"]["path"]
        hf_repo_qrels = (
            dataset_path + "-qrels" if "clarin-knext" in dataset_path else None
        )
        for split in kwargs.get("eval_splits", self.metadata_dict["eval_splits"]):
            (
                corpus,
                queries,
                og_relevant_docs,
                changed_relevant_docs,
                top_ranked_init,
            ) = HFDataLoaderInstructions(
                hf_repo=dataset_path,
                hf_repo_qrels=hf_repo_qrels,
                streaming=False,
                keep_in_memory=False,
            ).load(split=split)

            # Conversion from DataSet
            top_ranked = defaultdict(list)
            [
                top_ranked[cur_inst["qid"]].append(cur_inst["pid"])
                for cur_inst in top_ranked_init
            ]
            og_instructions = {
                query["text"]: query["instruction_og"] for query in queries
            }
            changed_instructions = {
                query["text"]: query["instruction_changed"] for query in queries
            }
            if self.do_length_ablation:
                keywords = {query["text"]: query["keywords"] for query in queries}
                short_instructions = {
                    query["text"]: query["short_query"] for query in queries
                }
            queries = {query["id"]: query["text"] for query in queries}
            corpus = {
                doc["id"]: {"title": doc["title"], "text": doc["text"]}
                for doc in corpus
            }
            assert len(top_ranked) == len(queries), (
                f"Top ranked not loaded properly! Expected {len(self.queries)} but got {len(self.top_ranked)}."
            )

            (
                self.corpus[split],
                self.queries[split],
                self.og_relevant_docs[split],
                self.changed_relevant_docs[split],
            ) = corpus, queries, og_relevant_docs, changed_relevant_docs
            self.changed_instructions[split], self.og_instructions[split] = (
                changed_instructions,
                og_instructions,
            )
            self.top_ranked[split] = top_ranked

            if self.do_length_ablation:
                self.keywords[split], self.short_instructions[split] = (
                    keywords,
                    short_instructions,
                )

        self.data_loaded = True

    def _evaluate_subset_lang(
        self,
        retriever: InstructionRetrievalEvaluator,
        corpus: dict,
        queries: dict,
        og_relevant_docs: dict,
        changed_relevant_docs: dict,
        og_instructions: dict,
        changed_instructions: dict,
        top_ranked: dict,
        lang: str,
        split: str,
        keywords: dict | None = None,
        short_instructions: dict | None = None,
        **kwargs,
    ) -> dict[str, dict[str, float] | float]:
        corpus, queries = corpus[split], queries[split]
        og_relevant_docs, changed_relevant_docs = (
            og_relevant_docs[split],
            changed_relevant_docs[split],
        )
        og_instructions, changed_instructions = (
            og_instructions[split],
            changed_instructions[split],
        )

        top_ranked = top_ranked[split]
        kwargs["prediction_name"] = "og"  # for naming predictions, as needed
        scores_og, results_og = self._evaluate_subset(
            retriever,
            corpus,
            queries,
            og_relevant_docs,
            og_instructions,
            top_ranked,
            lang,
            **kwargs,
        )
        kwargs["prediction_name"] = "changed"  # for naming predictions, as needed
        scores_changed, results_changed = self._evaluate_subset(
            retriever,
            corpus,
            queries,
            changed_relevant_docs,
            changed_instructions,
            top_ranked,
            lang,
            **kwargs,
        )

        newly_irrelevant_qrels = self.create_qrel_diff(
            og_relevant_docs,
            changed_relevant_docs,
        )
        overall_changed_scores = utils.evaluate_change(
            results_og, results_changed, newly_irrelevant_qrels
        )

        overall_changed_scores["individual"] = {
            "original": scores_og,
            "changed": scores_changed,
        }

        if self.do_length_ablation:
            keywords, short_instructions = (
                keywords[split],
                short_instructions[split],
            )
            kwargs["prediction_name"] = "base"  # for naming predictions, as needed
            scores_base, results_base = self._evaluate_subset(
                retriever,
                corpus,
                queries,
                og_relevant_docs,
                defaultdict(str),
                top_ranked,
                lang,
                **kwargs,
            )
            kwargs["prediction_name"] = "keywords"  # for naming predictions, as needed
            scores_w_keywords_scores, scores_w_keywords_results = self._evaluate_subset(
                retriever,
                corpus,
                queries,
                og_relevant_docs,
                keywords,
                top_ranked,
                lang,
                **kwargs,
            )
            kwargs["prediction_name"] = (
                "short_instr"  # for naming predictions, as needed
            )
            (
                scores_w_short_instr_scores,
                scores_w_short_instr_result,
            ) = self._evaluate_subset(
                retriever,
                corpus,
                queries,
                og_relevant_docs,
                short_instructions,
                top_ranked,
                lang,
                **kwargs,
            )
            overall_changed_scores["length_ablation"] = {
                "keywords": scores_w_keywords_scores,
                "short_instructions": scores_w_short_instr_scores,
                "base": scores_base,
            }

        return overall_changed_scores

    def evaluate(
        self,
        model: Encoder,
        split: str = "test",
        subsets_to_run: list[str] | None = None,
        *,
        encode_kwargs: dict[str, Any] = {},
        **kwargs,
    ) -> dict[str, dict[str, Any]]:
        retriever = InstructionRetrievalEvaluator(
            retriever=model,
            task_name=self.metadata.name,
            encode_kwargs=encode_kwargs,
            **kwargs,
        )
        scores = {}
        if self.is_multilingual:
            hf_subsets = self.hf_subsets
            if subsets_to_run is not None:
                hf_subsets = [s for s in hf_subsets if s in subsets_to_run]

            for lang in hf_subsets:
                logger.info(f"Language: {lang}")
                scores[lang] = self._evaluate_subset_lang(
                    retriever,
                    corpus=self.corpus[lang],
                    queries=self.queries[lang],
                    og_relevant_docs=self.og_relevant_docs[lang],
                    changed_relevant_docs=self.changed_relevant_docs[lang],
                    og_instructions=self.og_instructions[lang],
                    changed_instructions=self.changed_instructions[lang],
                    top_ranked=self.top_ranked[lang],
                    lang=lang,
                    split=split,
                    keywords=self.keywords[lang] if self.do_length_ablation else None,
                    short_instructions=self.short_instructions[lang]
                    if self.do_length_ablation
                    else None,
                    **kwargs,
                )
                self._add_main_score(scores[lang])
        else:
            lang = "default"
            scores[lang] = self._evaluate_subset_lang(
                retriever,
                corpus=self.corpus,
                queries=self.queries,
                og_relevant_docs=self.og_relevant_docs,
                changed_relevant_docs=self.changed_relevant_docs,
                og_instructions=self.og_instructions,
                changed_instructions=self.changed_instructions,
                top_ranked=self.top_ranked,
                lang=lang,
                split=split,
                keywords=self.keywords if self.do_length_ablation else None,
                short_instructions=self.short_instructions
                if self.do_length_ablation
                else None,
                **kwargs,
            )
            self._add_main_score(scores[lang])

        return scores

    def _add_main_score(self, scores: dict[str, dict[str, float]]) -> None:
        scores["main_score"] = scores[self.metadata.main_score]

    def _evaluate_subset(
        self,
        retriever: InstructionRetrievalEvaluator,
        corpus: dict[str, dict[str, str]],
        queries: dict[str, str],
        relevant_docs: dict[str, dict[str, int]],
        instructions: dict[str, str],
        top_ranked: dict[str, list[str]],
        lang=None,
        **kwargs,
    ) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
        start_time = time()

        # do the results by query and relevant docs only
        all_results = []
        for query_id in tqdm.tqdm(list(queries.keys()), leave=False, desc="Retrieving"):
            cur_queries = {query_id: queries[query_id]}
            cur_instructions = {queries[query_id]: instructions[queries[query_id]]}
            cur_docs = {
                key: value
                for (key, value) in corpus.items()
                if key in top_ranked[query_id]
            }
            all_results.append(
                retriever(
                    cur_docs, cur_queries, instructions=cur_instructions, qid=query_id
                )
            )

        # combine all the results (which are {'qid' -> {'doc_id' -> score} mappings)
        # we know all are unique qids, so we can smash together
        results = {k: v for d in all_results for k, v in d.items()}

        end_time = time()
        logger.info(f"Time taken to retrieve: {end_time - start_time:.2f} seconds")

        if kwargs.get("save_predictions", False):
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
                    f"{output_folder}/{self.metadata_dict['name']}_predictions.json"
                )
            else:
                qrels_save_path = f"{output_folder}/{self.metadata_dict['name']}_{lang}_predictions.json"

            if kwargs.get("prediction_name", None):
                qrels_save_path = qrels_save_path.replace(
                    ".json", f"_{kwargs['prediction_name']}.json"
                )

            with open(qrels_save_path, "w") as f:
                json.dump(results, f)

        ndcg, _map, recall, precision, naucs = retriever.evaluate(
            relevant_docs,
            results,
            retriever.k_values,
            ignore_identical_ids=kwargs.get("ignore_identical_ids", True),
        )
        mrr, naucs = retriever.evaluate_custom(
            relevant_docs, results, retriever.k_values, "mrr"
        )
        scores = {
            **{f"ndcg_at_{k.split('@')[1]}": v for (k, v) in ndcg.items()},
            **{f"map_at_{k.split('@')[1]}": v for (k, v) in _map.items()},
            **{f"recall_at_{k.split('@')[1]}": v for (k, v) in recall.items()},
            **{f"precision_at_{k.split('@')[1]}": v for (k, v) in precision.items()},
            **{f"naucs_at_{k.split('@')[1]}": v for (k, v) in naucs.items()},
            **{f"mrr_at_{k.split('@')[1]}": v for (k, v) in mrr.items()},
        }
        return scores, results

    def create_qrel_diff(self, og_qrels, changed_qrels):
        newly_irrelevant_qrels = {}
        for qid in og_qrels:
            newly_irrelevant_qrels[qid] = []
            for doc_id in og_qrels[qid]:
                if changed_qrels[qid][doc_id] != og_qrels[qid][doc_id]:
                    newly_irrelevant_qrels[qid].append(doc_id)

        return newly_irrelevant_qrels

    def _calculate_metrics_from_split(
        self, split: str, hf_subset: str | None = None, compute_overall: bool = False
    ) -> InstructionRetrievalDescriptiveStatistics:
        if hf_subset:
            corpus = self.corpus[hf_subset][split]
            queries = self.queries[hf_subset][split]
            relevant_docs = self.og_relevant_docs[hf_subset][split]
            og_instructions = self.og_instructions[hf_subset][split]
            changed_instructions = self.changed_instructions[hf_subset][split]
            top_ranked = self.top_ranked[hf_subset][split]
        elif compute_overall:
            corpus = {}
            queries = {}
            relevant_docs = {}
            og_instructions = {}
            changed_instructions = {}
            top_ranked = {}
            for hf_subset in self.metadata.eval_langs:
                corpus.update(process_docs(self.corpus, hf_subset, split))
                queries.update(process_docs(self.queries, hf_subset, split))
                relevant_docs.update(
                    process_relevant_docs(self.og_relevant_docs, hf_subset, split)
                )
                og_instructions.update(
                    process_docs(
                        self.og_instructions,
                        hf_subset,
                        split,
                    )
                )
                changed_instructions.update(
                    process_docs(self.changed_instructions, hf_subset, split)
                )
                top_ranked.update(process_top_ranked(self.top_ranked, hf_subset, split))
        else:
            corpus = self.corpus[split]
            queries = self.queries[split]
            relevant_docs = self.og_relevant_docs[split]
            og_instructions = self.og_instructions[split]
            changed_instructions = self.changed_instructions[split]
            top_ranked = self.top_ranked[split]

        corpus_combined = [
            doc.get("title", "") + doc["text"] for doc in corpus.values()
        ]
        corpus_len = [len(doc) for doc in corpus_combined]
        total_corpus_len = sum(corpus_len)

        queries_len = [len(query) for query in queries.values()]
        total_queries_len = sum(queries_len)
        instructions_len = [
            len(instruction) for instruction in og_instructions.values()
        ]
        total_instructions_len = sum(instructions_len)
        changed_instructions_len = [
            len(instruction) for instruction in changed_instructions.values()
        ]
        total_changed_instructions_len = sum(changed_instructions_len)
        qrels_non_zero = [
            sum(1 for doc_id in docs if docs[doc_id] != 0)
            for docs in relevant_docs.values()
        ]
        num_qrels_non_zero = sum(qrels_non_zero)
        qrels_per_doc = num_qrels_non_zero / len(relevant_docs) if len(queries) else 0
        ranked_per_query = [len(docs) for docs in top_ranked.values()]
        top_ranked_per_query = (
            sum(ranked_per_query) / len(queries) if len(queries) else 0
        )
        return InstructionRetrievalDescriptiveStatistics(
            num_samples=len(queries) + len(corpus),
            num_docs=len(corpus),
            num_queries=len(queries),
            number_of_characters=total_corpus_len
            + total_queries_len
            + total_instructions_len
            + total_changed_instructions_len,
            min_document_length=min(corpus_len),
            average_document_length=(
                total_corpus_len / len(corpus) if len(corpus) else 0
            ),
            max_document_length=max(corpus_len),
            unique_docs=len(set(corpus_combined)),
            min_query_length=min(queries_len),
            average_query_length=(
                total_queries_len / len(queries) if len(queries) else 0
            ),
            max_query_length=max(queries_len),
            unique_queries=len(set(queries.values())),
            min_instruction_length=min(instructions_len),
            average_instruction_length=(
                total_instructions_len / len(queries) if len(queries) else 0
            ),
            max_instruction_length=max(instructions_len),
            unique_instructions=len(set(og_instructions.values())),
            min_changed_instruction_length=min(changed_instructions_len),
            average_changed_instruction_length=(
                total_changed_instructions_len / len(queries) if len(queries) else 0
            ),
            max_changed_instruction_length=max(changed_instructions_len),
            unique_changed_instructions=len(set(changed_instructions.values())),
            min_average_relevant_docs_per_query=min(qrels_non_zero),
            average_relevant_docs_per_query=qrels_per_doc,
            max_average_relevant_docs_per_query=max(qrels_non_zero),
            min_average_top_ranked_per_query=min(ranked_per_query),
            average_top_ranked_per_query=top_ranked_per_query,
            max_average_top_ranked_per_query=max(ranked_per_query),
        )


def process_docs(
    collection: dict[str, dict[str, dict[str, str]]], hf_subset: str, split: str
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


def process_top_ranked(
    collection: dict[str, dict[str, dict[str, list[str]]]], hf_subset: str, split: str
) -> dict[str, list[str]]:
    return_collection = {}
    for query_id, docs_id in collection[hf_subset][split].items():
        return_collection[f"{split}_{hf_subset}_{query_id}"] = [
            f"{split}_{hf_subset}_{doc_id}" for doc_id in docs_id
        ]
    return return_collection
