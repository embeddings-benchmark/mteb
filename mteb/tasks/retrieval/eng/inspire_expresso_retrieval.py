from __future__ import annotations

from typing import Any

import pyarrow.parquet as pq
from datasets import Dataset
from huggingface_hub import hf_hub_download

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class INSPIREExpressoAT2ARetrieval(AbsTaskRetrieval):
    """Score all documents, filter by the released exclusions, then apply MTEB metrics.

    Equal scores follow trec_eval's document-ID tie break. The upstream script
    uses a seeded random tie break when ranking from embeddings.
    """

    metadata = TaskMetadata(
        name="INSPIREExpressoAT2ARetrieval",
        description=(
            "Retrieve human-recorded Expresso speech by following an instruction about "
            "the reference speaker, speaking style, or both. Each of 200 recordings "
            "has four instructions and each query has two relevant documents. The "
            "released exclusion lists reveal relevance when compared across queries, "
            "so the evaluator applies them after full-corpus scoring. Meta's Expresso "
            "recordings are CC BY-NC 4.0. INSPIRE specifies no separate license for "
            "its instructions, relevance judgments, or exclusions."
        ),
        reference="https://www.isca-archive.org/interspeech_2026/li26r_interspeech.html",
        dataset={
            "path": "lca0503/INSPIRE",
            "revision": "4202b965891ca79897ce8cb639e08c591e993c06",
        },
        type="Any2AnyRetrieval",
        category="at2a",
        modalities=["audio", "text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2026-01-13", "2026-02-04"),  # First and pinned INSPIRE data uploads.
        domains=["Spoken", "Speech"],
        task_subtypes=["Speech Retrieval"],
        annotations_creators="derived",
        sample_creation="created",
        dialect=[],
        license="not specified",
        prompt={
            "query": "Follow the instruction to retrieve speech matching the reference recording."
        },
        bibtex_citation=r"""
@inproceedings{li26r_interspeech,
  author = {Li, Chen-An and Lee, Hung-yi},
  booktitle = {Interspeech 2026},
  doi = {10.21437/Interspeech.2026-1026},
  pages = {7324--7333},
  title = {INSPIRE: A Benchmark for Instruction-Aware Speech Retrieval},
  url = {https://www.isca-archive.org/interspeech_2026/li26r_interspeech.html},
  year = {2026},
}
""",
    )
    k_values = (1, 5, 10, 20, 50, 100)
    filter_candidates_after_search = True
    # The stock cross-encoder wrapper requires a shortlist.
    _support_cross_encoder = False

    def load_data(self, num_proc: int | None = None, **kwargs: Any) -> None:
        if self.data_loaded:
            return

        def read_shard(kind: str, columns: list[str]) -> Dataset:
            path = hf_hub_download(
                repo_id=self.metadata.dataset["path"],
                revision=self.metadata.dataset["revision"],
                repo_type="dataset",
                filename=f"{kind}/Expresso-00000-of-00001.parquet",
            )
            return Dataset(pq.read_table(path, columns=columns))

        corpus = read_shard("document", ["audio_id", "audio"]).rename_column(
            "audio_id", "id"
        )
        rows = read_shard(
            "query", ["audio", "instruction", "positive_documents", "excluded_ids"]
        )
        # An upstream audio_id is shared by four instructions. Row order is pinned.
        query_ids = [f"q{i:05d}" for i in range(len(rows))]
        queries = rows.select_columns(["audio", "instruction"]).rename_column(
            "instruction", "text"
        )
        document_ids = list(corpus["id"])
        relevant_docs = {}
        candidates = {}
        judgments = rows.select_columns(["positive_documents", "excluded_ids"])
        for qid, row in zip(query_ids, judgments, strict=True):
            relevant_docs[qid] = dict.fromkeys(row["positive_documents"], 1)
            excluded = set(row["excluded_ids"])
            candidates[qid] = [doc for doc in document_ids if doc not in excluded]

        self.dataset = {
            "default": {
                "test": {
                    "corpus": corpus,
                    "queries": queries.add_column("id", query_ids),
                    "relevant_docs": relevant_docs,
                    "top_ranked": candidates,
                }
            }
        }
        self.data_loaded = True
