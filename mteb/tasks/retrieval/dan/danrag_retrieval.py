from __future__ import annotations

from typing import Any

from datasets import load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata
from mteb.timing import TimingStack

_SOURCE_REVISION = "408cf48d4b78182878f06eb445b20de201d58d74"

_CITATION = r"""
@misc{schmidt2025danragbench,
  author = {Johan Hausted Schmidt},
  institution = {IT University of Copenhagen},
  title = {DanRAG-Bench: A Danish Multimodal Document Retrieval Benchmark Across Five Sectors},
  year = {2025},
}
"""


def _load_danrag_data(
    task: AbsTaskRetrieval,
    include_text: bool,
    num_proc: int | None,
    timer: TimingStack | None = None,
) -> None:
    if task.data_loaded:
        return

    path = task.metadata.dataset["path"]
    revision = task.metadata.dataset["revision"]
    splits = task.metadata.eval_splits
    dataset: dict[str, dict[str, dict[str, Any]]] = {"default": {}}

    timer = timer or TimingStack()
    with timer("Data loading", log_message=f"Loading dataset {task.metadata.name}..."):
        for split in splits:
            corpus_columns = ["page_id", "image"]
            if include_text:
                corpus_columns.insert(1, "text")

            corpus = load_dataset(
                path,
                data_files={split: "corpus/*.parquet"},
                split=split,
                revision=revision,
                num_proc=num_proc,
            )
            corpus = corpus.select_columns(corpus_columns).rename_column(
                "page_id", "id"
            )

            raw_queries = load_dataset(
                path,
                data_files={split: "queries/*.parquet"},
                split=split,
                revision=revision,
                num_proc=num_proc,
            )
            queries = raw_queries.select_columns(["id", "query"]).rename_column(
                "query", "text"
            )
            relevant_docs = {
                query_id: dict.fromkeys(valid_pages, 1)
                for query_id, valid_pages in zip(
                    raw_queries["id"], raw_queries["valid_pages"], strict=True
                )
            }

            dataset["default"][split] = {
                "corpus": corpus,
                "queries": queries,
                "relevant_docs": relevant_docs,
                "top_ranked": None,
            }

        task.dataset = dataset
    with timer("Dataset transform"):
        task.dataset_transform(num_proc=num_proc)
    task.data_loaded = True


_COMMON_METADATA = {
    "reference": "https://huggingface.co/datasets/Johanschmidt/DanRAG-Bench",
    "dataset": {
        "path": "Johanschmidt/DanRAG-Bench",
        "revision": _SOURCE_REVISION,
    },
    "type": "DocumentUnderstanding",
    "eval_splits": ["test"],
    "eval_langs": ["dan-Latn"],
    "main_score": "ndcg_at_5",
    "date": ("2023-01-01", "2024-12-31"),
    "domains": ["Government", "Financial", "Medical", "Legal", "Non-fiction"],
    "task_subtypes": ["Image Text Retrieval"],
    "license": "mit",
    "annotations_creators": "LM-generated and reviewed",
    "dialect": [],
    "modalities": ["text", "image"],
    "sample_creation": "LM-generated and verified",
    "bibtex_citation": _CITATION,
    "contributed_by": "Johan Hausted Schmidt",
}


class DanRAGT2IRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="DanRAGT2IRetrieval",
        description="DanRAG-Bench evaluates Danish visual document retrieval over 349 page images from eight public-sector documents spanning energy, finance, health, law, and municipalities. Its 471 manually verified queries ask for factual information; the retrieval goal is to find every page containing evidence for each answer.",
        category="t2i",
        prompt={"query": "Find a screenshot that is relevant to the user's question."},
        **_COMMON_METADATA,
    )

    def load_data(
        self,
        num_proc: int | None = None,
        *,
        timer: TimingStack | None = None,
        **kwargs: Any,
    ) -> None:
        _load_danrag_data(self, include_text=False, num_proc=num_proc, timer=timer)


class DanRAGT2ITRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="DanRAGT2ITRetrieval",
        description="DanRAG-Bench evaluates Danish multimodal document retrieval over the extracted text and page images of 349 pages from eight public-sector documents spanning energy, finance, health, law, and municipalities. Its 471 manually verified queries ask for factual information; the retrieval goal is to find every page containing evidence for each answer.",
        category="t2it",
        prompt={
            "query": "Find a document page that is relevant to the user's question."
        },
        **_COMMON_METADATA,
    )

    def load_data(
        self,
        num_proc: int | None = None,
        *,
        timer: TimingStack | None = None,
        **kwargs: Any,
    ) -> None:
        _load_danrag_data(self, include_text=True, num_proc=num_proc, timer=timer)
