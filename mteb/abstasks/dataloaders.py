from __future__ import annotations

import logging
import os
from collections import defaultdict

from datasets import Features, Sequence, Value, get_dataset_config_names, load_dataset

logger = logging.getLogger(__name__)


# Adapted from https://github.com/beir-cellar/beir/blob/f062f038c4bfd19a8ca942a9910b1e0d218759d4/beir/datasets/data_loader_hf.py#L10
class HFDataLoader:
    """This dataloader handles the dataloading for retrieval-oriented tasks, including standard retrieval, reranking, and instruction-based variants of the above.

    If the `hf_repo` is provided, the dataloader will fetch the data from the HuggingFace hub. Otherwise, it will look for the data in the specified `data_folder`.

    Required files include the corpus, queries, and qrels files. Optionally, the dataloader can also load instructions and top-ranked (for reranking) files.
    """

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
        self.instructions = {}
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
            self.top_ranked_file = (
                os.path.join(data_folder, "top_ranked.jsonl")
                if data_folder
                else "top_ranked.jsonl"
            )
            self.top_ranked_file = (
                None
                if not os.path.exists(self.top_ranked_file)
                else self.top_ranked_file
            )
            self.instructions_file = (
                os.path.join(data_folder, "instructions.jsonl")
                if data_folder
                else "instructions.jsonl"
            )
            self.instructions_file = (
                None
                if not os.path.exists(self.instructions_file)
                else self.instructions_file
            )
        self.streaming = streaming
        self.keep_in_memory = keep_in_memory
        self.trust_remote_code = trust_remote_code

    @staticmethod
    def check(fIn: str, ext: str):
        if not os.path.exists(fIn):
            raise ValueError(f"File {fIn} not present! Please provide accurate file.")

        if not fIn.endswith(ext):
            raise ValueError(f"File {fIn} must be present with extension {ext}")

    def load(
        self, split: str = "test"
    ) -> tuple[
        dict[str, dict[str, str]],  # corpus
        dict[str, str | list[str]],  # queries
        dict[str, dict[str, int]],  # qrels/relevant_docs
        dict[str, str | list[str]],  # instructions (optional)
        dict[str, list[str]] | dict[str, dict[str, float]],  # top_ranked (optional)
    ]:
        if not self.hf_repo:
            self.qrels_file = os.path.join(self.qrels_folder, split + ".tsv")
            self.check(fIn=self.corpus_file, ext="jsonl")
            self.check(fIn=self.query_file, ext="jsonl")
            self.check(fIn=self.qrels_file, ext="tsv")
            if self.top_ranked_file:
                self.check(fIn=self.top_ranked_file, ext="jsonl")
            if self.instructions_file:
                self.check(fIn=self.instructions_file, ext="jsonl")
            configs = []
        else:
            configs = get_dataset_config_names(self.hf_repo)

        if not len(self.corpus):
            logger.info("Loading Corpus...")
            self._load_corpus()
            logger.info("Loaded %d %s Documents.", len(self.corpus), split.upper())
            logger.info("Doc Example: %s", self.corpus[0])

        if not len(self.queries):
            logger.info("Loading Queries...")
            self._load_queries()

        if "top_ranked" in configs or (not self.hf_repo and self.top_ranked_file):
            logger.info("Loading Top Ranked")
            self._load_top_ranked()
            logger.info(
                f"Top ranked loaded: {len(self.top_ranked) if self.top_ranked else 0}"
            )
        else:
            self.top_ranked = None

        if "instruction" in configs or (not self.hf_repo and self.instructions_file):
            logger.info("Loading Instructions")
            self._load_instructions()
            logger.info(
                f"Instructions loaded: {len(self.instructions) if self.instructions else 0}"
            )
        else:
            self.instructions = None

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

        return self.corpus, self.queries, self.qrels, self.instructions, self.top_ranked

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
                trust_remote_code=self.trust_remote_code,
            )
        else:
            corpus_ds = load_dataset(
                "json",
                data_files=self.corpus_file,
                streaming=self.streaming,
                keep_in_memory=self.keep_in_memory,
                trust_remote_code=self.trust_remote_code,
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
                trust_remote_code=self.trust_remote_code,
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
        qrels_ds = qrels_ds.cast(features)
        self.qrels = qrels_ds

    def _load_top_ranked(self):
        if self.hf_repo:
            top_ranked_ds = load_dataset(
                self.hf_repo,
                "top_ranked",
                keep_in_memory=self.keep_in_memory,
                streaming=self.streaming,
                trust_remote_code=self.trust_remote_code,
            )
        else:
            top_ranked_ds = load_dataset(
                "json",
                data_files=self.top_ranked_file,
                streaming=self.streaming,
                keep_in_memory=self.keep_in_memory,
            )
        top_ranked_ds = next(iter(top_ranked_ds.values()))  # get first split
        if (
            "query-id" in top_ranked_ds.column_names
            and "corpus-ids" in top_ranked_ds.column_names
        ):
            # is a {query-id: str, corpus-ids: list[str]} format
            top_ranked_ds = top_ranked_ds.cast_column("query-id", Value("string"))
            top_ranked_ds = top_ranked_ds.cast_column(
                "corpus-ids", Sequence(Value("string"))
            )
        else:
            # is a {"query-id": {"corpus-id": score}} format, let's change it
            top_ranked_ds = top_ranked_ds.map(
                lambda x: {"query-id": x["query-id"], "corpus-ids": list(x.keys())},
                remove_columns=[
                    col for col in top_ranked_ds.column_names if col != "query-id"
                ],
            )

        top_ranked_ds = top_ranked_ds.remove_columns(
            [
                col
                for col in top_ranked_ds.column_names
                if col not in ["query-id", "corpus-ids"]
            ]
        )
        self.top_ranked = top_ranked_ds

    def _load_instructions(self):
        if self.hf_repo:
            instructions_ds = load_dataset(
                self.hf_repo,
                "instruction",
                keep_in_memory=self.keep_in_memory,
                streaming=self.streaming,
                trust_remote_code=self.metadata_dict["dataset"].get(
                    "trust_remote_code", False
                ),
            )
        else:
            instructions_ds = load_dataset(
                "json",
                data_files=self.instructions_file,
                streaming=self.streaming,
                keep_in_memory=self.keep_in_memory,
            )
        instructions_ds = next(iter(instructions_ds.values()))
        instructions_ds = instructions_ds.cast_column("query-id", Value("string"))
        instructions_ds = instructions_ds.cast_column("instruction", Value("string"))
        instructions_ds = instructions_ds.remove_columns(
            [
                col
                for col in instructions_ds.column_names
                if col not in ["query-id", "instruction"]
            ]
        )
        self.instructions = instructions_ds
