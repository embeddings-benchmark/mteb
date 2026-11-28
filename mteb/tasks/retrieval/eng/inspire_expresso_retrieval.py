from __future__ import annotations

from hashlib import sha256
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from datasets import Audio, Dataset
from huggingface_hub import hf_hub_download

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


def _model_inputs(rows: pa.Table) -> tuple[Dataset, list[str]]:
    """Remove source filenames and order recordings by a content-derived ID."""
    audio = pc.struct_field(rows["audio"], "bytes")
    texts = (
        rows["instruction"].to_pylist() if "instruction" in rows.column_names else None
    )
    ids = []
    for index, recording in enumerate(audio):
        digest = sha256(recording.as_buffer())
        if texts is None:
            ids.append("d" + digest.hexdigest())
        else:
            digest.update(b"\0" + texts[index].encode("utf-8"))
            ids.append("q" + digest.hexdigest())
    if len(ids) != len(set(ids)):
        raise ValueError(
            "Expected unique Expresso documents and audio-instruction pairs."
        )

    inputs = pa.table({"id": ids, "audio": audio})
    if texts is not None:
        inputs = inputs.append_column("text", rows["instruction"])
    # Sort the Arrow table itself so the backing rows also have the public order.
    dataset = Dataset(inputs.sort_by("id")).cast_column("audio", Audio())
    return dataset, ids


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
            "so the evaluator applies them after full-corpus scoring. Content hashes "
            "determine IDs and row order. The loader removes source filenames. "
            "Meta published the recordings in 2023 under CC BY-NC 4.0. The dates "
            "estimate when INSPIRE created its instructions. INSPIRE specifies no "
            "separate license for the instructions, relevance judgments, or exclusions."
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
        # Bounds for the instructions: GPT-5.2's public release to the pinned data.
        date=("2025-12-11", "2026-02-04"),
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

        def read_shard(kind: str, columns: list[str]) -> pa.Table:
            path = hf_hub_download(
                repo_id=self.metadata.dataset["path"],
                revision=self.metadata.dataset["revision"],
                repo_type="dataset",
                filename=f"{kind}/Expresso-00000-of-00001.parquet",
            )
            return pq.read_table(path, columns=columns)

        documents = read_shard("document", ["audio_id", "audio"])
        corpus, ids = _model_inputs(documents)
        source_to_id = dict(zip(documents["audio_id"].to_pylist(), ids, strict=True))
        del documents
        rows = read_shard(
            "query", ["audio", "instruction", "positive_documents", "excluded_ids"]
        )
        queries, query_ids = _model_inputs(rows)
        document_ids = list(corpus["id"])
        relevant_docs = {}
        candidates = {}
        judgments = rows.select(["positive_documents", "excluded_ids"]).to_pylist()
        for qid, row in zip(query_ids, judgments, strict=True):
            relevant_docs[qid] = {
                source_to_id[doc]: 1 for doc in row["positive_documents"]
            }
            excluded = {source_to_id[doc] for doc in row["excluded_ids"]}
            candidates[qid] = [doc for doc in document_ids if doc not in excluded]

        self.dataset = {
            "default": {
                "test": {
                    "corpus": corpus,
                    "queries": queries,
                    "relevant_docs": relevant_docs,
                    "top_ranked": candidates,
                }
            }
        }
        self.data_loaded = True
