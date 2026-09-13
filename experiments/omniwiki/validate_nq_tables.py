"""Validate the local NQ-Tables adapter; write aggregates, never dataset copies.

Run from the repository root with ``python -m experiments.omniwiki.validate_nq_tables
--cache-dir /path/to/datasets/cache``. Downloads use the adapter's pinned source.
"""

# This standalone validation script intentionally uses assertions, progress
# output, and MTEB's internal quality helpers rather than reimplementing checks.
# ruff: noqa: S101, T201, PLC2701

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import torch
from datasets import load_dataset

import mteb
from mteb._create_dataloaders import _corpus_to_dict
from mteb.models.search_wrappers import SearchEncoderWrapper
from mteb.tasks.retrieval.eng.nq_tables_retrieval import NQTablesRetrieval
from tests.test_tasks.test_task_quality import _WARNING_CHECK_KINDS, _split_quality

if TYPE_CHECKING:
    from collections.abc import Iterable


def _id_counts(ids: list[str]) -> dict[str, int]:
    counts = Counter(ids)
    return {
        "rows": len(ids),
        "unique_ids": len(counts),
        "duplicated_ids": sum(n > 1 for n in counts.values()),
        "duplicate_rows_beyond_first": sum(n - 1 for n in counts.values()),
    }


def _digest(rows: Iterable[dict[str, Any]]) -> str:
    """Hash ordered JSON records with UTF-8 strings and unambiguous framing."""
    digest = hashlib.sha256()
    for row in rows:
        payload = json.dumps(row, ensure_ascii=False, sort_keys=True).encode("utf-8")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _json_safe(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    return value


def main() -> None:  # noqa: PLR0914
    """Measure the complete release and evaluate the existing random baseline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/omniwiki/nqtables_validation"),
    )
    parser.add_argument("--skip-baseline", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "source": dict(NQTablesRetrieval.metadata.dataset),
        "license": None,
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "python": platform.python_version(),
        "mteb_module_path": str(Path(mteb.__file__).resolve()),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ["mteb", "datasets", "numpy", "torch", "pytest"]
        },
        "splits": {},
    }

    task = NQTablesRetrieval(seed=42).filter_eval_splits(["train", "dev", "test"])
    task.load_data(cache_dir=str(args.cache_dir))
    source_options = {**task.metadata.dataset, "cache_dir": str(args.cache_dir)}
    corpus = task.dataset["default"]["test"]["corpus"]
    corpus_ids = set(corpus["id"])
    source_corpus = load_dataset(**source_options, name="corpus_md", split="corpus_md")
    source_digest = _digest(source_corpus)
    adapted_digest = _digest(
        {"_id": row["id"], "title": row["title"], "text": row["text"]} for row in corpus
    )
    assert source_digest == adapted_digest, "Corpus content changed"
    report["corpus"] = {
        **_id_counts(list(corpus["id"])),
        "empty_texts": sum(not text.strip() for text in corpus["text"]),
        "empty_titles": sum(not title for title in corpus["title"]),
        "source_sha256": source_digest,
        "adapted_sha256": adapted_digest,
        "raw_text_characters": sum(len(text) for text in corpus["text"]),
        "cache_files": [item["filename"] for item in source_corpus.cache_files],
        "mteb_model_input_whitespace_changes": sum(
            _corpus_to_dict(row)["text"] != row["text"] for row in corpus
        ),
    }
    del source_corpus
    assert all(data["corpus"] is corpus for data in task.dataset["default"].values())
    report["shared_corpus_identity"] = True
    query_id_sets = {}
    for split, data in task.dataset["default"].items():
        queries, qrels = data["queries"], task.source_qrels[split]
        qids = set(queries["id"])
        query_id_sets[split] = qids
        source_queries = load_dataset(
            **source_options, name="queries", split=f"{split}_queries"
        )
        assert queries.rename_column("id", "_id").to_list() == source_queries.to_list()
        triples = Counter((row["qid"], row["did"], row["score"]) for row in qrels)
        mapped_triples = Counter(
            (qid, did, score)
            for qid, docs in data["relevant_docs"].items()
            for did, score in docs.items()
        )
        assert triples == mapped_triples, f"Qrels changed in {split}"
        report["splits"][split] = {
            "queries": {
                **_id_counts(list(queries["id"])),
                "empty_texts": sum(not text.strip() for text in queries["text"]),
                "conflicting_ids": sum(
                    len({row["text"] for row in queries if row["id"] == qid}) > 1
                    for qid, count in Counter(queries["id"]).items()
                    if count > 1
                ),
                "source_sha256": _digest(source_queries),
                "adapted_sha256": _digest(queries.rename_column("id", "_id")),
            },
            "corpus_items": len(corpus),
            "qrel_rows": len(qrels),
            "qrel_source_sha256": _digest(qrels),
            "qrel_triples_preserved": triples == mapped_triples,
            "query_cache_files": [
                item["filename"] for item in source_queries.cache_files
            ],
            "qrel_cache_files": [item["filename"] for item in qrels.cache_files],
            "duplicate_qrel_pairs": len(qrels)
            - len({(row["qid"], row["did"]) for row in qrels}),
            "scores": dict(Counter(row["score"] for row in qrels)),
            "positive_docs_per_unique_query": dict(
                Counter(
                    sum(
                        score > 0
                        for score in data["relevant_docs"].get(qid, {}).values()
                    )
                    for qid in qids
                )
            ),
            "queries_without_qrels": len(qids - data["relevant_docs"].keys()),
            "dangling_query_qrel_rows": sum(row["qid"] not in qids for row in qrels),
            "dangling_corpus_qrel_rows": sum(
                row["did"] not in corpus_ids for row in qrels
            ),
            "dangling_corpus_unique_ids": len(
                {row["did"] for row in qrels if row["did"] not in corpus_ids}
            ),
        }
        print(split, json.dumps(report["splits"][split]), flush=True)
    report["cross_split_query_id_overlap"] = {
        f"{left}/{right}": len(query_id_sets[left] & query_id_sets[right])
        for left, right in [("train", "dev"), ("train", "test"), ("dev", "test")]
    }

    # A measured regression check for the previously traced source example.
    qid = "dev_3052797144690241914_0_0"
    did = "Italian_general_election,_2018_B19E32FB82D4E3D7"
    test = task.dataset["default"]["test"]
    assert [row["text"] for row in test["queries"] if row["id"] == qid] == [
        "who is new prime minister of italy 2018"
    ]
    assert test["relevant_docs"][qid][did] == 1
    traced_text = next(row["text"] for row in corpus if row["id"] == did)
    assert hashlib.sha256(traced_text.encode("utf-8")).hexdigest() == (
        "329109192b73a661d39027e5ddc7a78ceddaa1843167f5fbc5e4e465921046ba"
    )
    report["traced_example_preserved"] = True

    # Use MTEB's actual per-split statistics and quality functions. Store stats
    # locally because this task is not registered for publication.
    stats = {}
    findings = []
    for split in task.supported_splits:
        stats[split] = task._calculate_descriptive_statistics_from_split(split)
        for check, message in _split_quality(task.metadata.name, split, stats[split]):
            findings.append(
                {
                    "split": split,
                    "check": check,
                    "message": message,
                    "severity": "warning"
                    if check.split(":", 1)[0] in _WARNING_CHECK_KINDS
                    else "error",
                }
            )
    report["quality_findings"] = findings
    report["quality_checks_passed"] = not any(
        finding["severity"] == "error" for finding in findings
    )
    task.metadata._validate_metadata()
    report["metadata_schema_valid"] = True
    report["publication_metadata_complete"] = task.metadata.is_filled()
    report["unset_metadata_fields"] = [
        name
        for name in type(task.metadata).model_fields
        if getattr(task.metadata, name) is None
    ]
    (args.output_dir / "descriptive_stats.json").write_text(
        json.dumps(stats, indent=4) + "\n"
    )

    if not args.skip_baseline:
        torch.set_num_threads(2)
        model = mteb.get_model("mteb/baseline-random-encoder", embed_dim=32)
        wrapper = SearchEncoderWrapper(model, corpus_chunk_size=10_000)
        baseline = {
            "model": model.mteb_model_meta.name,
            "revision": model.mteb_model_meta.revision,
            "embedding_dimension": 32,
            "seed": 42,
            "similarity": "cosine",
            "split": "test",
            "query_rows": len(test["queries"]),
            "unique_queries": len(set(test["queries"]["id"])),
            "corpus_items": len(corpus),
            "corpus_chunk_size": 10_000,
            "batch_size": 256,
            "torch_threads": 2,
            "requested_top_k": task._top_k,
        }
        search = wrapper.search

        def observe_search(*args: Any, **kwargs: Any) -> dict[str, dict[str, float]]:
            results = search(*args, **kwargs)
            baseline["result_query_ids"] = len(results)
            baseline["retrieved_docs_per_query"] = dict(
                Counter(len(docs) for docs in results.values())
            )
            return results

        start = time.monotonic()
        with patch.object(wrapper, "search", side_effect=observe_search):
            baseline["scores"] = task.evaluate(
                wrapper, split="test", encode_kwargs={"batch_size": 256}
            )
        baseline["elapsed_seconds"] = time.monotonic() - start
        baseline["nonfinite_metric_handling"] = (
            "Nonfinite metrics are serialized as null."
        )
        report["baseline"] = baseline
        # Evaluation must not change the original qrels or source query rows.
        assert (
            _digest(test["queries"].rename_column("id", "_id"))
            == report["splits"]["test"]["queries"]["source_sha256"]
        )
        assert Counter(
            (qid, did, score)
            for qid, docs in test["relevant_docs"].items()
            for did, score in docs.items()
        ) == Counter(
            (row["qid"], row["did"], row["score"]) for row in task.source_qrels["test"]
        )
        baseline["query_rows_and_qrels_preserved_after_evaluation"] = True

    (args.output_dir / "validation.json").write_text(
        json.dumps(_json_safe(report), indent=4, allow_nan=False) + "\n"
    )
    print("Validation report:", args.output_dir / "validation.json", flush=True)
    print("Quality findings:", json.dumps(findings), flush=True)


if __name__ == "__main__":
    main()
