"""STARD's released development split; no inferred or generated judgments."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from datasets import Dataset, DownloadManager

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

STARD_REVISION = "bae10e1f358685446c3af4b15e7b94724f83bc9e"
STARD_FILES = {
    "corpus.jsonl": "15c77a1333d947ec88cf4cb4c99d70dafd946ec91d35f2bbc77824817c7f7fa4",
    "example/dev.query.txt": "590e207b804c2829504040f14d7b4c19a64ba78e5f9e91f904fa2755d1ab47c4",
    "example/qrels.dev.tsv": "3f2cab03a9672b3f1179d7452819a985404a2cafb264f2cf6e4d0b1744b94ba2",
}


def _map_stard(corpus_path: Path, queries_path: Path, qrels_path: Path) -> dict:
    """Preserve source order/text; disjoint ID namespaces prevent self-hit removal."""
    corpus, queries, qrels = [], [], {}
    corpus_ids, query_ids = set(), set()
    for line in corpus_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        doc_id = f"d:{row['id']}"
        if doc_id in corpus_ids:
            raise ValueError(f"Duplicate corpus ID: {doc_id}")
        corpus_ids.add(doc_id)
        corpus.append({"id": doc_id, "title": row["name"], "text": row["content"]})
    for line in queries_path.read_text(encoding="utf-8").splitlines():
        source_id, text = line.split("\t", 1)
        query_id = f"q:{source_id}"
        if query_id in query_ids:
            raise ValueError(f"Duplicate query ID: {query_id}")
        query_ids.add(query_id)
        queries.append({"id": query_id, "text": text})
        qrels[query_id] = {}
    for line in qrels_path.read_text(encoding="utf-8").splitlines():
        source_qid, source_did = line.split("\t")
        query_id, doc_id = f"q:{source_qid}", f"d:{source_did}"
        if query_id not in query_ids or doc_id not in corpus_ids:
            raise ValueError(f"Dangling relevance pair: {query_id}, {doc_id}")
        if doc_id in qrels[query_id]:
            raise ValueError(f"Duplicate relevance pair: {query_id}, {doc_id}")
        qrels[query_id][doc_id] = 1
    if not corpus or not queries or any(not rels for rels in qrels.values()):
        raise ValueError("Empty corpus, queries, or query without positive judgments")
    return {
        "corpus": Dataset.from_list(corpus),
        "queries": Dataset.from_list(queries),
        "relevant_docs": qrels,
        "top_ranked": None,
    }


class STARDRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="STARDRetrieval",
        description="Chinese layperson legal-consultation queries retrieving statutory articles. Uses the author's released dev split and full corpus; binary human judgments are unchanged. Actual collection dates are unknown; the date range is a documented publication-year proxy (2024).",
        reference="https://aclanthology.org/2024.findings-emnlp.625/",
        dataset={
            "path": "https://github.com/oneal2000/STARD",
            "revision": STARD_REVISION,
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["dev"],
        eval_langs=["cmn-Hans"],
        main_score="ndcg_at_10",
        # Actual collection dates are unknown; use a publication-year proxy.
        # https://github.com/embeddings-benchmark/mteb/issues/5819#issuecomment-6538640893
        date=("2024-01-01", "2024-12-31"),
        domains=["Legal", "Written"],
        task_subtypes=["Article retrieval"],
        license="mit",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        prompt={
            "query": "Given a Chinese legal consultation question, retrieve relevant statutory articles."
        },
        bibtex_citation=r"""
@inproceedings{su-etal-2024-stard,
  title = {{STARD}: A Chinese Statute Retrieval Dataset Derived from Real-life Queries by Non-professionals},
  author = {Su, Weihang and Hu, Yiran and Xie, Anzhe and Ai, Qingyao and Bing, Quezi and Zheng, Ning and Liu, Yun and Shen, Weixing and Liu, Yiqun},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2024},
  year = {2024},
  doi = {10.18653/v1/2024.findings-emnlp.625},
  url = {https://aclanthology.org/2024.findings-emnlp.625/}
}
""",
    )

    def load_data(self, num_proc: int | None = None, **kwargs: Any) -> None:
        if self.data_loaded:
            return
        source_dir = kwargs.get("source_dir")
        if source_dir is None:
            base = f"https://raw.githubusercontent.com/oneal2000/STARD/{STARD_REVISION}/data"
            paths = DownloadManager().download(
                {name: f"{base}/{name}" for name in STARD_FILES}
            )
        else:
            paths = {name: str(Path(source_dir) / name) for name in STARD_FILES}
        for name, expected in STARD_FILES.items():
            actual = hashlib.sha256(Path(paths[name]).read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError(f"STARD SHA-256 mismatch for {name}: {actual}")
        self.dataset = {
            "default": {"dev": _map_stard(*(Path(paths[name]) for name in STARD_FILES))}
        }
        self.data_loaded = True
