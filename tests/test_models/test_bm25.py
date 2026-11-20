import pytest

import mteb


@pytest.mark.parametrize("model_name", ["mteb/baseline-bm25s", "mteb/baseline-bb25"])
def test_bm25_e2e(model_name):
    # fails for dataset smaller than 1000
    pytest.importorskip("bm25s", reason="bm25s not installed")
    pytest.importorskip("Stemmer", reason="PyStemmer not installed")

    model = mteb.get_model(model_name)
    task = mteb.get_task("NFCorpus", eval_splits=["test"])

    results = mteb.evaluate(model, task, cache=None)
    result = results[0]
    assert result.scores["test"][0]["ndcg_at_1"] == 0.42879


@pytest.mark.parametrize("model_name", ["mteb/baseline-bm25s", "mteb/baseline-bb25"])
def test_bm25_reranking_keeps_candidates_outside_corpus_top_k(model_name):
    pytest.importorskip("bm25s", reason="bm25s not installed")
    pytest.importorskip("Stemmer", reason="PyStemmer not installed")
    from datasets import Dataset

    from mteb.mocks.mock_tasks import MockRerankingTask

    texts = [
        "apple banana",
        "apple banana banana",
        "banana apple apple",
        "apple tart",
        "kiwi melon",
    ]
    corpus = Dataset.from_dict(
        {
            "id": [f"d{i}" for i in range(len(texts))],
            "text": texts,
        }
    )
    queries = Dataset.from_dict({"id": ["q1"], "text": ["apple banana"]})
    # neither candidate is in the full-corpus top 2 (d0, d1, d2 score higher)
    top_ranked = {"q1": ["d3", "d4"]}
    metadata = MockRerankingTask().metadata
    kwargs = {
        "task_metadata": metadata,
        "hf_split": "test",
        "hf_subset": "default",
        "encode_kwargs": {},
    }

    model = mteb.get_model(model_name)
    model.index(corpus, **kwargs)
    results = model.search(queries, top_k=2, top_ranked=top_ranked, **kwargs)

    assert set(results["q1"]) == {"d3", "d4"}
    assert results["q1"]["d3"] > results["q1"]["d4"]

    # An empty candidate mapping still requests reranking, not full retrieval.
    assert model.search(queries, top_k=2, top_ranked={}, **kwargs) == {"q1": {}}
