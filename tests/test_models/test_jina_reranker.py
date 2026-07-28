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
