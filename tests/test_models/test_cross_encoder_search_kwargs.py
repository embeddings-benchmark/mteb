from types import SimpleNamespace

import numpy as np
import pytest
from datasets import Dataset

from mteb.mocks.mock_tasks import MockRetrievalTask
from mteb.models.search_wrappers import SearchCrossEncoderWrapper
from mteb.models.sentence_transformer_wrapper import CrossEncoderWrapper


@pytest.mark.parametrize("batch_size", [None, 1, 3])
def test_search_forwards_inference_batch_size(batch_size):
    """DataLoader batching must not replace the requested model inference batch size."""
    calls = []
    requested_batch_size = batch_size
    omitted = object()

    # Some cross-encoders accept batch_size but no other inference kwargs.
    def predict(
        inputs1,
        inputs2,
        *,
        task_metadata,
        hf_split,
        hf_subset,
        batch_size=omitted,
    ):
        calls.append(batch_size)
        assert inputs1.batch_size == inputs2.batch_size == (requested_batch_size or 32)
        return np.array([0.75, 0.25])

    model = SimpleNamespace(predict=predict, mteb_model_meta=None)
    wrapper = SearchCrossEncoderWrapper(model)
    metadata = MockRetrievalTask.metadata
    options = {"show_progress_bar": False, "precision": "float32"}
    if batch_size is not None:
        options["batch_size"] = batch_size
    wrapper.index(
        Dataset.from_dict({"id": ["d1", "d2"], "text": ["relevant", "other"]}),
        task_metadata=metadata,
        hf_split="test",
        hf_subset="default",
        encode_kwargs=options,
    )
    results = wrapper.search(
        Dataset.from_dict({"id": ["q1"], "text": ["question"]}),
        task_metadata=metadata,
        hf_split="test",
        hf_subset="default",
        top_k=2,
        top_ranked={"q1": ["d1", "d2"]},
        encode_kwargs=options,
    )
    assert calls == [omitted if batch_size is None else batch_size]
    assert results == {"q1": {"d1": 0.75, "d2": 0.25}}


def test_search_controls_sentence_transformers_inference_batches(tmp_path):
    """Check actual inference batches and score ordering without downloading weights."""
    from sentence_transformers import CrossEncoder
    from transformers import AutoTokenizer, BertConfig, BertForSequenceClassification

    vocabulary = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "query",
        "other",
        "document",
    ]
    (tmp_path / "vocab.txt").write_text("\n".join(vocabulary), encoding="utf-8")
    config = BertConfig(
        vocab_size=len(vocabulary),
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
        num_labels=1,
    )
    BertForSequenceClassification(config).save_pretrained(tmp_path)
    tokenizer = AutoTokenizer.from_pretrained(tmp_path, local_files_only=True)
    tokenizer.save_pretrained(tmp_path)
    backend = CrossEncoder(str(tmp_path), device="cpu")
    wrapper = SearchCrossEncoderWrapper(CrossEncoderWrapper(backend))

    documents = ["document " * n for n in [2, 7, 1, 5, 3, 6, 4]]
    metadata = MockRetrievalTask.metadata
    options = {"batch_size": 3}
    wrapper.index(
        Dataset.from_dict({"id": [f"d{i}" for i in range(7)], "text": documents}),
        task_metadata=metadata,
        hf_split="test",
        hf_subset="default",
        encode_kwargs=options,
    )
    batch_sizes = []

    def record_batch(module, args):
        batch_sizes.append(args[0].shape[0])

    handle = backend.model.get_input_embeddings().register_forward_pre_hook(
        record_batch
    )
    try:
        result = wrapper.search(
            Dataset.from_dict({"id": ["q1", "q2"], "text": ["query", "other query"]}),
            task_metadata=metadata,
            hf_split="test",
            hf_subset="default",
            top_k=4,
            top_ranked={"q2": ["d5", "d1", "d3"], "q1": ["d6", "d0", "d4", "d2"]},
            encode_kwargs=options,
        )
    finally:
        handle.remove()

    assert batch_sizes == [3, 3, 1]
    # Use a different batch size as the reference and check each query-document pair.
    expected_ids = [("q2", i) for i in [5, 1, 3]] + [("q1", i) for i in [6, 0, 4, 2]]
    pairs = [
        ("query" if query_id == "q1" else "other query", documents[i])
        for query_id, i in expected_ids
    ]
    expected = backend.predict(pairs, batch_size=1, show_progress_bar=False)
    actual = [result[query_id][f"d{i}"] for query_id, i in expected_ids]
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
