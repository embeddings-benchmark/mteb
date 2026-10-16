"""Execute the offline retrieval-downsampling documentation examples."""

import re
from pathlib import Path

import pytest


@pytest.fixture
def example():
    page = Path(__file__).parents[1] / "docs/contributing/downsampling.md"
    blocks = re.findall(r"```python\n(.*?)```", page.read_text(), re.DOTALL)
    namespace = {}
    for block in blocks[:2]:
        exec(compile(block, str(page), "exec"), namespace)  # noqa: S102 - execute repository-owned documentation
    return namespace


def test_query_only_and_input_unchanged(example):
    data = example["data"]
    sample = example["sample_retrieval"](data, n_queries=2)
    assert len(sample["corpus"]) == len(data["corpus"])
    assert len(data["queries"]) == 3
    assert len(data["relevant_docs"]) == 3
    qid = next(iter(sample["relevant_docs"]))
    sample["relevant_docs"][qid].clear()
    assert data["relevant_docs"][qid]


def test_selection_independent_of_row_order(example):
    data = example["data"]
    reversed_data = {
        **data,
        "queries": data["queries"].select([2, 1, 0]),
        "corpus": data["corpus"].select([5, 4, 3, 2, 1, 0]),
    }
    sample = example["sample_retrieval"](reversed_data, n_queries=2, corpus_size=4)
    assert sample["queries"].to_dict() == example["sample"]["queries"].to_dict()
    assert sample["corpus"].to_dict() == example["sample"]["corpus"].to_dict()


@pytest.mark.parametrize(
    ("n_queries", "corpus_size"), [(0, None), (4, None), (3, 1), (2, 7)]
)
def test_invalid_sizes(example, n_queries, corpus_size):
    with pytest.raises(ValueError, match="n_queries|corpus_size"):
        example["sample_retrieval"](
            example["data"], n_queries=n_queries, corpus_size=corpus_size
        )


@pytest.mark.parametrize("judgments", [{}, {"d1": 0}, {"missing": 1}])
def test_invalid_judgments(example, judgments):
    data = example["data"]
    data["relevant_docs"]["q1"] = judgments
    with pytest.raises(ValueError, match="no positive judgment|missing documents"):
        example["sample_retrieval"](data, n_queries=3)


def test_rejects_candidate_lists(example):
    data = {**example["data"], "top_ranked": {"q1": ["d1"]}}
    with pytest.raises(ValueError, match="candidate-restricted"):
        example["sample_retrieval"](data, n_queries=2)


def test_preserves_zero_and_graded_judgments(example):
    sample = example["sample_retrieval"](example["data"], n_queries=3, corpus_size=4)
    assert sample["relevant_docs"]["q1"] == {"d1": 2, "d4": 0}
    assert set(sample["corpus"]["id"]) == {"d1", "d2", "d3", "d4"}


def test_rejects_duplicate_ids(example):
    data = example["data"]
    data["queries"] = data["queries"].select([0, 0, 1])
    with pytest.raises(ValueError, match="unique string IDs"):
        example["sample_retrieval"](data, n_queries=2)
