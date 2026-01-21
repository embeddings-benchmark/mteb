from __future__ import annotations

import logging
from typing import TYPE_CHECKING, TypedDict

from datasets import (
    Dataset,
    Features,
    Sequence,
    Value,
    get_dataset_config_names,
    get_dataset_split_names,
    load_dataset,
)

if TYPE_CHECKING:
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

    Attributes:
        corpus: The corpus dataset containing documents. Should have columns `id`, `title`, `text` or `image`.
        queries: The queries dataset containing queries. Should have columns `id`, `text`, `instruction` (for instruction retrieval/reranking) or `image`.
        relevant_docs: A mapping of query IDs to relevant document IDs and their relevance scores. Should have columns `query-id`, `corpus-id`, `score`.
        top_ranked: A mapping of query IDs to a list of top-ranked document IDs. Should have columns `query-id`, `corpus-ids` (list[str]). This is optional and used for reranking tasks.
    """

    corpus: CorpusDatasetType
    queries: QueryDatasetType
    relevant_docs: RelevantDocumentsType
    top_ranked: TopRankedDocumentsType | None


class RetrievalDatasetLoader:
    """This dataloader handles the dataloading for retrieval-oriented tasks, including standard retrieval, reranking, and instruction-based variants.

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
        """Initializes the dataloader with the specified parameters.

        Args:
            hf_repo: The HuggingFace repository name or path.
            revision: The specific revision of the dataset to use.
            trust_remote_code: Passes the `trust_remote_code` argument to `load_dataset`.
            split: The dataset split to load (e.g., "train", "validation", "test").
            config: The specific configuration of the dataset to use. If None, the default configuration is used.

        Warning: Deprecated
            trust_remote_code is deprecated and will be removed in future versions. Please ensure that the datasets you are using do not require remote code execution.
        """
        self.revision = revision
        self.hf_repo = hf_repo
        self.trust_remote_code = trust_remote_code
        self.split = split
        self.config = config if config != "default" else None
        self.dataset_configs = get_dataset_config_names(self.hf_repo, self.revision)

    def load(
        self,
        num_proc: int = 1,
    ) -> RetrievalSplitData:
        """Loads the dataset split for the specified configuration.

        Args:
            num_proc: The number of processes to use.

        Returns:
            A dictionary containing the corpus, queries, relevant documents, instructions (if applicable), and top-ranked documents (if applicable).
        """
        top_ranked = None

        qrels = self._load_qrels(num_proc)
        corpus = self._load_corpus(num_proc)
        queries = self._load_queries(num_proc)

        queries = queries.filter(
            lambda x: x["id"] in qrels.keys(), desc="Filtering queries by qrels"
        )

        if any(c.endswith("top_ranked") for c in self.dataset_configs):
            top_ranked = self._load_top_ranked(num_proc)

        if any(c.endswith("instruction") for c in self.dataset_configs):
            instructions = self._load_instructions(num_proc)
            queries = _combine_queries_with_instructions_datasets(
                queries, instructions, num_proc
            )

        return RetrievalSplitData(
            corpus=corpus,
            queries=queries,
            relevant_docs=qrels,
            top_ranked=top_ranked,
        )

    def _get_split(self, config: str) -> str:
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

    def _load_dataset_split(self, config: str, num_proc: int) -> Dataset:
        return load_dataset(
            self.hf_repo,
            config,
            split=self._get_split(config),
            trust_remote_code=self.trust_remote_code,
            revision=self.revision,
            num_proc=num_proc,
        )

    def _load_corpus(self, num_proc: int) -> CorpusDatasetType:
        config = f"{self.config}-corpus" if self.config is not None else "corpus"
        logger.info("Loading corpus subset: %s", config)

        corpus_ds = self._load_dataset_split(config, num_proc)
        if "_id" in corpus_ds.column_names:
            corpus_ds = corpus_ds.cast_column("_id", Value("string")).rename_column(
                "_id", "id"
            )
        logger.info("Loaded %d %s Documents.", len(corpus_ds), self.split.upper())
        logger.debug("Doc Example: %s", corpus_ds[0])
        return corpus_ds

    def _load_queries(self, num_proc: int) -> QueryDatasetType:
        config = f"{self.config}-queries" if self.config is not None else "queries"
        logger.info("Loading queries subset: %s", config)

        if "query" in self.dataset_configs:
            config = "query"
        queries_ds = self._load_dataset_split(config, num_proc)
        if "_id" in queries_ds.column_names:
            queries_ds = queries_ds.cast_column("_id", Value("string")).rename_column(
                "_id", "id"
            )

        logger.info("Loaded %d %s queries.", len(queries_ds), self.split.upper())
        logger.debug("Query Example: %s", queries_ds[0])

        return queries_ds

    def _load_qrels(self, num_proc: int) -> RelevantDocumentsType:
        config = f"{self.config}-qrels" if self.config is not None else "default"

        logger.info("Loading qrels subset: %s", config)
        if config == "default" and config not in self.dataset_configs:
            if "qrels" in self.dataset_configs:
                config = "qrels"
            else:
                raise ValueError(
                    "No qrels or default config found. Please specify a valid config or ensure the dataset has qrels."
                )

        qrels_ds = self._load_dataset_split(config, num_proc)
        qrels_ds = qrels_ds.select_columns(["query-id", "corpus-id", "score"])

        qrels_ds = qrels_ds.cast(
            Features(
                {
                    "query-id": Value("string"),
                    "corpus-id": Value("string"),
                    "score": Value("int32"),
                }
            )
        )

        qrels_ds = qrels_ds.to_polars()
        # filter queries with no qrels
        qrels_dict = {
            query_id[0]: dict(zip(group["corpus-id"], group["score"]))
            for query_id, group in qrels_ds.group_by("query-id", maintain_order=False)
        }

        logger.info("Loaded %d %s qrels.", len(qrels_dict), self.split.upper())
        return qrels_dict

    def _load_top_ranked(self, num_proc: int) -> TopRankedDocumentsType:
        config = (
            f"{self.config}-top_ranked" if self.config is not None else "top_ranked"
        )
        logger.info("Loading top ranked subset: %s", config)
        top_ranked_ds = self._load_dataset_split(config, num_proc)
        top_ranked_ds = top_ranked_ds.cast(
            Features(
                {
                    "query-id": Value("string"),
                    "corpus-ids": Sequence(Value("string")),
                }
            )
        ).select_columns(["query-id", "corpus-ids"])

        top_ranked_ds = top_ranked_ds.to_polars()

        queries = top_ranked_ds["query-id"].to_list()
        corpus_lists = top_ranked_ds["corpus-ids"].to_list()
        top_ranked_dict = dict(zip(queries, corpus_lists))
        logger.info(f"Top ranked loaded: {len(top_ranked_ds)}")
        return top_ranked_dict

    def _load_instructions(self, num_proc: int) -> InstructionDatasetType:
        config = (
            f"{self.config}-instruction" if self.config is not None else "instruction"
        )
        logger.info("Loading instruction subset: %s", config)
        instructions_ds = self._load_dataset_split(config, num_proc)
        instructions_ds = instructions_ds.cast(
            Features(
                {
                    "query-id": Value("string"),
                    "instruction": Value("string"),
                }
            )
        ).select_columns(["query-id", "instruction"])
        return instructions_ds


def _combine_queries_with_instructions_datasets(
    queries_dataset: QueryDatasetType,
    instruction_dataset: InstructionDatasetType | dict[str, str],
    num_proc: int,
) -> Dataset:
    if isinstance(instruction_dataset, Dataset):
        instruction_to_query_idx = {
            row["query-id"]: row["instruction"] for row in instruction_dataset
        }
    else:
        instruction_to_query_idx = instruction_dataset

    def _add_instruction_to_query(row: dict[str, str]) -> dict[str, str]:
        row["instruction"] = instruction_to_query_idx[row["id"]]
        return row

    return queries_dataset.map(_add_instruction_to_query, num_proc=num_proc)
