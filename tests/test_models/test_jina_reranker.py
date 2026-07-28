from unittest.mock import MagicMock

from mteb.models.model_implementations.jina_models import JinaRerankerV3Wrapper


def test_jina_reranker_passes_revision(monkeypatch):
    model = MagicMock()
    from_pretrained = MagicMock(return_value=model)
    monkeypatch.setattr("transformers.AutoModel.from_pretrained", from_pretrained)

    JinaRerankerV3Wrapper(
        "jinaai/jina-reranker-v3.5",
        revision="test-revision",
        device="cpu",
    )

    from_pretrained.assert_called_once_with(
        "jinaai/jina-reranker-v3.5",
        revision="test-revision",
        trust_remote_code=True,
        dtype="auto",
    )
    model.to.assert_called_once_with("cpu")
    model.eval.assert_called_once_with()


def test_jina_reranker_v35_predict_uses_supported_arguments():
    class JinaRerankerV35:
        @staticmethod
        def rerank(query, documents, top_n=None, return_embeddings=False):
            assert query == "query"
            assert documents == ["first", "second"]
            return [
                {"index": 1, "relevance_score": -0.25},
                {"index": 0, "relevance_score": 0.75},
            ]

    wrapper = object.__new__(JinaRerankerV3Wrapper)
    wrapper.model = JinaRerankerV35()

    scores = wrapper.predict(
        [{"text": ["query", "query"]}],
        [{"text": ["first", "second"]}],
        task_metadata=MagicMock(),
        hf_split="test",
        hf_subset="default",
    )

    assert scores.tolist() == [0.75, -0.25]


def test_jina_reranker_v3_predict_preserves_length_arguments():
    class JinaRerankerV3:
        @staticmethod
        def rerank(
            query,
            documents,
            max_query_length=None,
            max_doc_length=None,
        ):
            assert max_query_length == 3072
            assert max_doc_length == 2048
            return [{"index": 0, "relevance_score": 0.5}]

    wrapper = object.__new__(JinaRerankerV3Wrapper)
    wrapper.model = JinaRerankerV3()

    scores = wrapper.predict(
        [{"text": ["query"]}],
        [{"text": ["document"]}],
        task_metadata=MagicMock(),
        hf_split="test",
        hf_subset="default",
    )

    assert scores.tolist() == [0.5]
