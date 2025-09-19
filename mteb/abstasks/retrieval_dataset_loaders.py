from __future__ import annotations

import logging
from collections import defaultdict
from typing import TypedDict

from datasets import (
    Dataset,
    Features,
    Sequence,
    Value,
    get_dataset_config_names,
    get_dataset_split_names,
    load_dataset,
)

from mteb.types import (
    CorpusDatasetType,
    InstructionDatasetType,
    QueryDatasetType,
    RelevantDocumentsType,
    TopRankedDocumentsType,
)

logger = logging.getLogger(__name__)


class RetrievalSplitData(TypedDict):
    """A dictionary containing the corpus, queries, relevant documents, instructions, and top-ranked documents for a retrieval task.

    - `corpus`: A mapping of document IDs to their text or title and text. Columns: "id", "title", "text"
    - `queries`: A mapping of query IDs to their text. Columns: "id", "text" (can be `Conversation`)
    - `relevant_docs`: A mapping of query IDs to a mapping of document IDs and their relevance scores.
    - `instructions`: A mapping of query IDs to their instructions (if applicable). Columns: "query-id", "instruction"
    - `top_ranked`: A mapping of query IDs to a list of top-ranked document IDs (if applicable). Columns: "query-id", "corpus-ids"
    """

    corpus: CorpusDatasetType
    queries: QueryDatasetType
    relevant_docs: RelevantDocumentsType
    top_ranked: TopRankedDocumentsType | None


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
        self.dataset_configs = get_dataset_config_names(self.hf_repo, self.revision)

    def load(self) -> RetrievalSplitData:
        top_ranked = None

        qrels = self._load_qrels()
        corpus = self._load_corpus()
        queries = self._load_queries()

        queries = queries.filter(
            lambda x: x["id"] in qrels.keys(), desc="Filtering queries by qrels"
        )

        if any(c.endswith("top_ranked") for c in self.dataset_configs):
            top_ranked = self._load_top_ranked()

        if any(c.endswith("instruction") for c in self.dataset_configs):
            instructions = self._load_instructions()
            queries = combine_queries_with_instructions_datasets(queries, instructions)

        return RetrievalSplitData(
            corpus=corpus,
            queries=queries,
            relevant_docs=qrels,
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

    def _load_corpus(self) -> CorpusDatasetType:
        logger.info("Loading Corpus...")

        config = f"{self.config}-corpus" if self.config is not None else "corpus"
        corpus_ds = self.load_dataset_split(config)
        if "_id" in corpus_ds.column_names:
            corpus_ds = corpus_ds.cast_column("_id", Value("string")).rename_column(
                "_id", "id"
            )
        logger.info("Loaded %d %s Documents.", len(corpus_ds), self.split.upper())
        logger.info("Doc Example: %s", corpus_ds[0])
        return corpus_ds

    def _load_queries(self) -> QueryDatasetType:
        logger.info("Loading Queries...")

        config = f"{self.config}-queries" if self.config is not None else "queries"
        if "query" in self.dataset_configs:
            config = "query"
        queries_ds = self.load_dataset_split(config)
        if "_id" in queries_ds.column_names:
            queries_ds = queries_ds.cast_column("_id", Value("string")).rename_column(
                "_id", "id"
            )

        logger.info("Loaded %d %s queries.", len(queries_ds), self.split.upper())
        logger.info("Query Example: %s", queries_ds[0])

        return queries_ds

    def _load_qrels(self) -> RelevantDocumentsType:
        logger.info("Loading qrels...")

        config = f"{self.config}-qrels" if self.config is not None else "default"
        if config == "default" and config not in self.dataset_configs:
            if "qrels" in self.dataset_configs:
                config = "qrels"
            else:
                raise ValueError(
                    "No qrels or default config found. Please specify a valid config or ensure the dataset has qrels."
                )

        qrels_ds = self.load_dataset_split(config)
        qrels_ds = qrels_ds.select_columns(["query-id", "corpus-id", "score"])

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
        qrels_dict: dict[str, dict[str, int]] = defaultdict(dict)

        def qrels_dict_init(row):
            qrels_dict[row["query-id"]][row["corpus-id"]] = int(row["score"])

        qrels_ds.map(qrels_dict_init, desc="Creating qrels dict")
        logger.info("Loaded %d %s qrels.", len(qrels_dict), self.split.upper())
        return qrels_dict

    def _load_top_ranked(self) -> TopRankedDocumentsType:
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

        top_ranked_dict = {tr["query-id"]: tr["corpus-ids"] for tr in top_ranked_ds}
        logger.info(f"Top ranked loaded: {len(top_ranked_ds) if top_ranked_ds else 0}")
        return top_ranked_dict

    def _load_instructions(self) -> InstructionDatasetType:
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
        return instructions_ds


def combine_queries_with_instructions_datasets(
    queries_dataset: QueryDatasetType,
    instruction_dataset: InstructionDatasetType | dict[str, str],
) -> Dataset:
    if isinstance(instruction_dataset, Dataset):
        instruction_to_query_idx = {
            row["query-id"]: row["instruction"] for row in instruction_dataset
        }
    else:
        instruction_to_query_idx = instruction_dataset

    def add_instruction_to_query(row: dict[str, str]) -> dict[str, str]:
        row["instruction"] = instruction_to_query_idx[row["id"]]
        return row

    return queries_dataset.map(add_instruction_to_query)
