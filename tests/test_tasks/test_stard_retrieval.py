"""Synthetic fixtures below test plumbing only, never benchmark gold labels."""

import json
import math

import pytest

from mteb._evaluators.retrieval_metrics import calculate_retrieval_scores
from mteb.tasks.retrieval.zho.stard_retrieval import STARDRetrieval, _map_stard


def files(tmp_path):
    corpus = tmp_path / "corpus"
    queries = tmp_path / "queries"
    qrels = tmp_path / "qrels"
    corpus.write_text(
        "\n".join(
            json.dumps(r)
            for r in [
                {"id": 1, "name": "甲", "content": "相同正文"},
                {"id": 2, "name": "乙", "content": "相同正文"},
            ]
        ),
        encoding="utf-8",
    )
    queries.write_text("1\t问题？\n", encoding="utf-8")
    qrels.write_text("1\t2\n", encoding="utf-8")
    return corpus, queries, qrels


def test_mapping_preserves_titles_and_distinct_ids(tmp_path):
    data = _map_stard(*files(tmp_path))
    assert data["corpus"]["id"] == ["d:1", "d:2"]
    assert data["queries"][0] == {"id": "q:1", "text": "问题？"}
    assert data["corpus"]["title"] == ["甲", "乙"]
    assert data["relevant_docs"] == {"q:1": {"d:2": 1}}
    assert data["top_ranked"] is None  # {} incorrectly selects empty reranking


@pytest.mark.parametrize(
    ("rows", "error"),
    [
        ("1\t9\n", "Dangling"),
        ("9\t1\n", "Dangling"),
        ("1\t2\n1\t2\n", "Duplicate relevance"),
        ("", "without positive"),
    ],
)
def test_bad_qrels_fail_closed(tmp_path, rows, error):
    paths = files(tmp_path)
    paths[2].write_text(rows)
    with pytest.raises(ValueError, match=error):
        _map_stard(*paths)


def test_duplicate_ids_rejected(tmp_path):
    paths = files(tmp_path)
    paths[1].write_text("1\tx\n1\ty\n")
    with pytest.raises(ValueError, match="Duplicate query"):
        _map_stard(*paths)


def test_checksum_rejects_changed_source(tmp_path):
    (tmp_path / "corpus.jsonl").write_text("tampered")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        STARDRetrieval().load_data(source_dir=tmp_path)


def test_official_metrics_nontrivial_rank():
    # Two artificial queries; one relevant hit at rank 2, one entirely missed.
    qrels = {"q:a": {"d:a": 1, "d:b": 1}, "q:b": {"d:c": 1}}
    run = {"q:a": {"d:x": 2.0, "d:a": 1.0}, "q:b": {"d:y": 2.0, "d:z": 1.0}}
    scores = calculate_retrieval_scores(run, qrels, [10, 100])
    expected = (1 / math.log2(3)) / (1 + 1 / math.log2(3)) / 2
    assert scores.ndcg["NDCG@10"] == round(expected, 5)
    assert scores.recall["Recall@10"] == 0.25
