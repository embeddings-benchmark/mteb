import json
import logging
import os
from collections import defaultdict
from time import time
from typing import Dict, List, Tuple

import tqdm
from datasets import Features, Value, load_dataset

from ..evaluation.evaluators import utils
from ..evaluation.evaluators.InstructionRetrievalEvaluator import (
    InstructionRetrievalEvaluator,
)
from .AbsTask import AbsTask
from .AbsTaskRetrieval import HFDataLoader

logger = logging.getLogger(__name__)


# Adapted from https://github.com/beir-cellar/beir/blob/f062f038c4bfd19a8ca942a9910b1e0d218759d4/beir/datasets/data_loader_hf.py#L10
class HFDataLoaderInstructions(HFDataLoader):
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

    def load(
        self, split="test"
    ) -> Tuple[
        Dict[str, Dict[str, str]],
        Dict[str, str],
        Dict[str, Dict[str, int]],
        Dict[str, Dict[str, int]],
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

    def load_top_ranked(self) -> Dict[str, Dict[str, str]]:
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


class AbsTaskInstructionRetrieval(AbsTask):
    """Abstract class for retrieval tasks that use instructions. An example from Core17 would be
        query: What is the ongoing status of The Three Gorges Project?
        instruction: A relevant document will provide the projected or actual date of completion of the project, its estimated or actual total cost, or the estimated or ongoing electrical output of the finished project. Discussions of the social, political, or ecological impact of the project are not relevant.

    Child-classes must implement the following properties:
    self.corpus = Dict[id, Dict[str, str]] #id => dict with document datas like title and text
    self.queries = Dict[id, str] #id => query
    self.relevant_docs = List[id, id, score]
    self.og_instructions = Dict[str, str] query => original instruction
    self.changed_instructions = Dict[str, str] query => changed instruction
    self.top_ranked = Dict[id, List[id]] #id => list of top ranked document ids

    See https://arxiv.org/abs/2403.15246 for more details
    """

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
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.do_length_ablation = kwargs.get("do_length_ablation", False)

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
            corpus, queries, og_qrels, changed_qrels, top_ranked_init = (
                HFDataLoaderInstructions(
                    hf_repo=dataset_path,
                    hf_repo_qrels=hf_repo_qrels,
                    streaming=False,
                    keep_in_memory=False,
                ).load(split=split)
            )

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
            queries = {query["id"]: query["text"] for query in queries}
            corpus = {
                doc["id"]: {"title": doc["title"], "text": doc["text"]}
                for doc in corpus
            }
            if self.do_length_ablation:
                keywords = {query["text"]: query["keywords"] for query in queries}
                short_instructions = {
                    query["text"]: query["short_query"] for query in queries
                }
            assert (
                len(top_ranked) == len(queries)
            ), f"Top ranked not loaded properly! Expected {len(self.queries)} but got {len(self.top_ranked)}."

            (
                self.corpus[split],
                self.queries[split],
                self.og_relevant_docs[split],
                self.changed_relevant_docs[split],
            ) = corpus, queries, og_qrels, changed_qrels
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

    def evaluate(self, model, split="test", **kwargs):
        retriever = InstructionRetrievalEvaluator(model, **kwargs)

        scores_og = {}
        scores_changed = {}
        results_og = {}
        results_changed = {}
        scores_base = {}
        results_base = {}
        overall_changed_scores = {}
        if self.is_multilingual:
            for lang in self.langs:
                logger.info(f"Language: {lang}")
                corpus, queries = self.corpus[lang][split], self.queries[lang][split]
                og_relevant_docs, changed_relevant_docs = (
                    self.og_relevant_docs[lang][split],
                    self.changed_relevant_docs[lang][split],
                )
                og_instructions, changed_instructions = (
                    self.og_instructions[lang][split],
                    self.changed_instructions[lang][split],
                )

                top_ranked = self.top_ranked[lang][split]
                scores_og[lang], results_og[lang] = self._evaluate_monolingual(
                    retriever,
                    corpus,
                    queries,
                    og_relevant_docs,
                    "og",
                    og_instructions,
                    top_ranked,
                    lang,
                    **kwargs,
                )
                scores_changed[lang], results_changed[lang] = (
                    self._evaluate_monolingual(
                        retriever,
                        corpus,
                        queries,
                        changed_relevant_docs,
                        "changed",
                        changed_instructions,
                        top_ranked,
                        lang,
                        **kwargs,
                    )
                )

                newly_irrelevant_qrels = self.create_qrel_diff(
                    self.og_relevant_docs[lang][split],
                    self.changed_relevant_docs[lang][split],
                )
                overall_changed_scores[lang] = utils.evaluate_change(
                    results_og[lang], results_changed[lang], newly_irrelevant_qrels
                )

                overall_changed_scores[lang]["individual"] = {
                    "original": scores_og[lang],
                    "changed": scores_changed[lang],
                }

                if self.do_length_ablation:
                    keywords, short_instructions = (
                        self.keywords[lang][split],
                        self.short_instructions[lang][split],
                    )
                    scores_base[lang], results_base[lang] = self._evaluate_monolingual(
                        retriever,
                        corpus,
                        queries,
                        og_relevant_docs,
                        "base",
                        defaultdict(str),
                        top_ranked,
                        lang,
                        **kwargs,
                    )
                    scores_w_keywords = self._evaluate_monolingual(
                        retriever,
                        corpus,
                        queries,
                        og_relevant_docs,
                        "keywords",
                        keywords,
                        top_ranked,
                        lang,
                        **kwargs,
                    )
                    scores_w_short_instr = self._evaluate_monolingual(
                        retriever,
                        corpus,
                        queries,
                        og_relevant_docs,
                        "short_instructions",
                        short_instructions,
                        top_ranked,
                        lang,
                        **kwargs,
                    )
                    overall_changed_scores[lang]["length_ablation"] = {
                        "keywords": scores_w_keywords,
                        "short_instructions": scores_w_short_instr,
                        "base": scores_base[lang],
                    }
        else:
            corpus, queries = self.corpus[split], self.queries[split]
            og_relevant_docs, changed_relevant_docs = (
                self.og_relevant_docs[split],
                self.changed_relevant_docs[split],
            )
            og_instructions, changed_instructions = (
                self.og_instructions[split],
                self.changed_instructions[split],
            )
            top_ranked = self.top_ranked[split]

            scores_og, results_og = self._evaluate_monolingual(
                retriever,
                corpus,
                queries,
                og_relevant_docs,
                "og",
                og_instructions,
                top_ranked,
                None,
                **kwargs,
            )

            scores_changed, results_changed = self._evaluate_monolingual(
                retriever,
                corpus,
                queries,
                changed_relevant_docs,
                "changed",
                changed_instructions,
                top_ranked,
                None,
                **kwargs,
            )

            newly_irrelevant_qrels = self.create_qrel_diff(
                self.og_relevant_docs[split], self.changed_relevant_docs[split]
            )
            overall_changed_scores = utils.evaluate_change(
                results_og, results_changed, newly_irrelevant_qrels
            )

            overall_changed_scores["individual"] = {
                "original": scores_og,
                "changed": scores_changed,
                "base": scores_base,
            }

            if self.do_length_ablation:
                keywords, short_instructions = (
                    self.keywords[split],
                    self.short_instructions[split],
                )
                scores_w_keywords = self._evaluate_monolingual(
                    retriever,
                    corpus,
                    queries,
                    og_relevant_docs,
                    "keywords",
                    keywords,
                    top_ranked,
                    None,
                    **kwargs,
                )
                scores_w_short_instr = self._evaluate_monolingual(
                    retriever,
                    corpus,
                    queries,
                    og_relevant_docs,
                    "short_instructions",
                    short_instructions,
                    top_ranked,
                    None,
                    **kwargs,
                )
                scores_base, results_base = self._evaluate_monolingual(
                    retriever,
                    corpus,
                    queries,
                    og_relevant_docs,
                    "base",
                    defaultdict(str),
                    top_ranked,
                    None,
                    **kwargs,
                )
                overall_changed_scores["length_ablation"] = {
                    "keywords": scores_w_keywords,
                    "short_instructions": scores_w_short_instr,
                    "base": scores_base,
                }

        return overall_changed_scores

    def _evaluate_monolingual(
        self,
        retriever: InstructionRetrievalEvaluator,
        corpus: Dict[str, Dict[str, str]],
        queries: Dict[str, str],
        relevant_docs: Dict[str, Dict[str, int]],
        qrels_name: str,
        instructions: Dict[str, str],
        top_ranked: Dict[str, List[str]],
        lang=None,
        **kwargs,
    ):
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
        logger.info(
            "Time taken to retrieve: {:.2f} seconds".format(end_time - start_time)
        )

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
        return scores, results

    def create_qrel_diff(self, og_qrels, changed_qrels):
        newly_irrelevant_qrels = {}
        for qid in og_qrels:
            newly_irrelevant_qrels[qid] = []
            for doc_id in og_qrels[qid]:
                if changed_qrels[qid][doc_id] != og_qrels[qid][doc_id]:
                    newly_irrelevant_qrels[qid].append(doc_id)

        return newly_irrelevant_qrels

    def calculate_metadata_metrics(self) -> None:
        self.load_data()

        for split in self.metadata_dict["eval_splits"]:
            if self.is_multilingual:
                for lang in self.og_relevant_docs.keys():
                    process_language(
                        self.og_relevant_docs[lang][split],
                        self.queries[lang][split],
                        self.corpus[lang][split],
                        self.changed_instructions[lang][split],
                        lang,
                    )
            else:
                process_language(
                    self.og_relevant_docs[split],
                    self.queries[split],
                    self.corpus[split],
                    self.changed_instructions[split],
                )


def process_language(relevant_docs, queries, corpus, instructions, lang=None):
    total_length, num_pairs = calculate_length_and_count(
        relevant_docs, queries, corpus, instructions
    )
    average_length = total_length / num_pairs if num_pairs else 0
    num_documents = len(queries) + len(corpus)

    language_description = f" for language {lang}" if lang else ""
    print(
        f"Average character length for changed{language_description} is {average_length}"
    )
    print(
        f"Number of queries and documents{language_description} is {num_documents} (repeated 2x)"
    )


def calculate_length_and_count(relevant_docs, queries, corpus, instructions):
    total_length = 0
    num_pairs = 0
    for query_id, docs in relevant_docs.items():
        query = queries[query_id]
        query += " " + instructions[query]
        for doc_id in docs:
            doc = corpus[doc_id]
            doc_text = doc["title"] + doc["text"]
            total_length += len(query) + len(doc_text)
            num_pairs += 1
    return total_length, num_pairs
