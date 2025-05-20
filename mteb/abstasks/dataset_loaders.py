from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Mapping
from typing import TypedDict

from datasets import (
    Dataset,
    DatasetDict,
    Features,
    Sequence,
    Value,
    get_dataset_config_names,
    get_dataset_split_names,
    load_dataset,
)

logger = logging.getLogger(__name__)


class RetrievalSplitData(TypedDict):
    """A dictionary containing the corpus, queries, relevant documents, instructions, and top-ranked documents for a retrieval task.
    - `corpus`: A mapping of document IDs to their text or title and text.
    - `queries`: A mapping of query IDs to their text.
    - `relevant_docs`: A mapping of query IDs to a mapping of document IDs and their relevance scores.
    - `instructions`: A mapping of query IDs to their instructions (if applicable).
    - `top_ranked`: A mapping of query IDs to a list of top-ranked document IDs (if applicable).
    """

    corpus: Mapping[str, str | dict[str, str]]
    queries: Mapping[str, str]
    relevant_docs: Mapping[str, Mapping[str, float]]
    instructions: Mapping[str, str] | None
    top_ranked: Mapping[str, list[str]] | None


class RetrievalDatasetLoader:
    """This dataloader handles the dataloading for retrieval-oriented tasks, including standard retrieval, reranking, and instruction-based variants of the above.

    If the `hf_repo` is provided, the dataloader will fetch the data from the HuggingFace hub. Otherwise, it will look for the data in the specified `data_folder`.

    Required files include the corpus, queries, and qrels files. Optionally, the dataloader can also load instructions and top-ranked (for reranking) files.
    """

    def __init__(
        self,
        hf_repo: str,
        revision: str,
        trust_remote_code: bool = False,
        split: str = "test",
        config: str | None = None,
    ):
        self.revision = revision
        self.hf_repo = hf_repo
        self.trust_remote_code = trust_remote_code
        self.split = split
        self.config = config if config != "default" else None

    def load(self) -> RetrievalSplitData:
        top_ranked = None
        instructions = None

        configs = get_dataset_config_names(
            self.hf_repo, self.revision, trust_remote_code=self.trust_remote_code
        )
        qrels = self._load_qrels()
        corpus = self._load_corpus()
        queries = self._load_queries()

        queries = {
            query["id"]: query["text"]
            for query in queries.filter(lambda x: x["id"] in qrels)
        }

        if any(c.endswith("top_ranked") for c in configs):
            top_ranked = self._load_top_ranked()

        if any(c.endswith("instruction") for c in configs):
            instructions = self._load_instructions()

        return RetrievalSplitData(
            corpus=corpus,
            queries=queries,
            relevant_docs=qrels,
            instructions=instructions,
            top_ranked=top_ranked,
        )

    def get_split(self, config: str) -> str:
        splits = get_dataset_split_names(
            self.hf_repo,
            revision=self.revision,
            config_name=config,
        )
        if self.split in splits:
            return self.split
        if len(splits) == 1:
            return splits[0]
        raise ValueError(
            f"Split {self.split} not found in {splits}. Please specify a valid split."
        )

    def load_dataset_split(self, config: str) -> Dataset:
        return load_dataset(
            self.hf_repo,
            config,
            split=self.get_split(config),
            trust_remote_code=self.trust_remote_code,
            revision=self.revision,
        )

    def _load_corpus(self) -> dict[str, dict[str, str]]:
        logger.info("Loading Corpus...")

        config = f"{self.config}-corpus" if self.config is not None else "corpus"
        corpus_ds = self.load_dataset_split(config)
        corpus_ds = corpus_ds.cast_column("_id", Value("string")).rename_column(
            "_id", "id"
        )
        logger.info("Loaded %d %s Documents.", len(corpus_ds), self.split.upper())
        logger.info("Doc Example: %s", corpus_ds[0])
        return {
            doc["id"]: {
                "title": doc.get("title", ""),
                "text": doc["text"],
            }
            for doc in corpus_ds
        }

    def _load_queries(self) -> Dataset | DatasetDict:
        logger.info("Loading Queries...")

        config = f"{self.config}-queries" if self.config is not None else "queries"
        queries_ds = self.load_dataset_split(config)
        queries_ds = (
            queries_ds.cast_column("_id", Value("string"))
            .rename_column("_id", "id")
            .select_columns(["id", "text"])
        )
        logger.info("Loaded %d %s queries.", len(queries_ds), self.split.upper())
        logger.info("Query Example: %s", queries_ds[0])

        return queries_ds

    def _load_qrels(self) -> dict[str, dict[str, float]]:
        logger.info("Loading qrels...")

        config = f"{self.config}-qrels" if self.config is not None else "default"

        qrels_ds = self.load_dataset_split(config)

        qrels_ds = qrels_ds.cast(
            Features(
                {
                    "query-id": Value("string"),
                    "corpus-id": Value("string"),
                    "score": Value("float"),
                }
            )
        )

        # filter queries with no qrels
        qrels_dict = defaultdict(dict)

        def qrels_dict_init(row):
            qrels_dict[row["query-id"]][row["corpus-id"]] = int(row["score"])

        qrels_ds.map(qrels_dict_init)
        logger.info("Loaded %d %s qrels.", len(qrels_dict), self.split.upper())
        return qrels_dict

    def _load_top_ranked(self) -> dict[str, str]:
        logger.info("Loading Top Ranked")

        config = (
            f"{self.config}-top_ranked" if self.config is not None else "top_ranked"
        )
        top_ranked_ds = self.load_dataset_split(config)
        top_ranked_ds = top_ranked_ds.cast(
            Features(
                {
                    "query-id": Value("string"),
                    "corpus-ids": Sequence(Value("string")),
                }
            )
        ).select_columns(["query-id", "corpus-ids"])

        top_ranked_ds = {tr["query-id"]: tr["corpus-ids"] for tr in top_ranked_ds}
        logger.info(f"Top ranked loaded: {len(top_ranked_ds) if top_ranked_ds else 0}")
        return top_ranked_ds

    def _load_instructions(self) -> dict[str, str]:
        logger.info("Loading Instructions")

        config = (
            f"{self.config}-instruction" if self.config is not None else "instruction"
        )
        instructions_ds = self.load_dataset_split(config)
        instructions_ds = instructions_ds.cast(
            Features(
                {
                    "query-id": Value("string"),
                    "instruction": Value("string"),
                }
            )
        ).select_columns(["query-id", "instruction"])

        instructions_ds = {
            row["query-id"]: row["instruction"] for row in instructions_ds
        }
        logger.info(
            f"Instructions loaded: {len(instructions_ds) if instructions_ds else 0}"
        )
        return instructions_ds
