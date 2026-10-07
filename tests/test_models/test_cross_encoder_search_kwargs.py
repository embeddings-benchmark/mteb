from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from datasets import Dataset

from mteb.mocks.mock_tasks import MockRetrievalTask
from mteb.models.search_wrappers import SearchCrossEncoderWrapper


@pytest.mark.parametrize("batch_size", [None, 1, 3])
def test_search_forwards_prediction_options(batch_size):
    """DataLoader batching must not replace the requested model inference batch size."""
    calls = []
    requested_batch_size = batch_size
    omitted = object()

    def predict(
        inputs1,
        inputs2,
        *,
        task_metadata,
        hf_split,
        hf_subset,
        batch_size=omitted,
        **kwargs: Any,
    ):
        calls.append(batch_size)
        assert kwargs == {"show_progress_bar": False, "precision": "float32"}
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
    original_options = options.copy()
    results = wrapper.search(
        Dataset.from_dict({"id": ["q1"], "text": ["question"]}),
        task_metadata=metadata,
        hf_split="test",
        hf_subset="default",
        top_k=2,
        top_ranked={"q1": ["d1", "d2"]},
        encode_kwargs=options,
    )
    assert options == original_options
    assert calls == [omitted if batch_size is None else batch_size]
    assert results == {"q1": {"d1": 0.75, "d2": 0.25}}


@pytest.mark.parametrize(
    "precision", [None, "float32", "int8", "uint8", "binary", "ubinary"]
)
@pytest.mark.parametrize("adapter", ["sentence_transformers", "kalm", "qwen3"])
def test_cross_encoder_adapters_handle_precision(adapter, precision, monkeypatch):
    import warnings

    from torch.utils.data import DataLoader

    from mteb.models.model_implementations.kalm_reranker import KaLMRerankerWrapper
    from mteb.models.model_implementations.qwen3_reranker import Qwen3RerankerWrapper
    from mteb.models.sentence_transformer_wrapper import CrossEncoderWrapper

    calls = []
    expected = np.array([0.25, 0.75], dtype=np.float32)

    if adapter == "sentence_transformers":
        wrapper = CrossEncoderWrapper.__new__(CrossEncoderWrapper)
        wrapper.query_prefix = wrapper.passage_prefix = ""
        for name in [
            "fps",
            "max_frames",
            "num_frames",
            "target_sampling_rate",
            "max_samples",
        ]:
            setattr(wrapper, name, None)

        # A strict backend proves embedding precision is consumed by the adapter.
        def predict(pairs, *, batch_size, show_progress_bar):
            assert pairs == [("query", "short"), ("query", "longer document")]
            assert batch_size == 1
            assert show_progress_bar is False
            calls.append(len(pairs))
            return expected

        wrapper.model = SimpleNamespace(predict=predict)
    elif adapter == "kalm":
        wrapper = KaLMRerankerWrapper.__new__(KaLMRerankerWrapper)
        wrapper.instruction = "instruction"

        def predict_batch(pairs, instructions):
            calls.append(len(pairs))
            return [0.25 if doc == "short" else 0.75 for _, doc in pairs]

        monkeypatch.setattr(wrapper, "_predict_batch", predict_batch)
    else:
        wrapper = Qwen3RerankerWrapper.__new__(Qwen3RerankerWrapper)
        monkeypatch.setattr(
            wrapper, "format_instruction", lambda instr, query, doc: doc
        )
        monkeypatch.setattr(wrapper, "process_inputs", lambda pairs: pairs)

        def compute_logits(pairs):
            calls.append(len(pairs))
            return [0.25 if doc == "short" else 0.75 for doc in pairs]

        monkeypatch.setattr(wrapper, "compute_logits", compute_logits)

    options = {"batch_size": 1, "show_progress_bar": False}
    if precision is not None:
        options["precision"] = precision
    original_options = options.copy()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = wrapper.predict(
            DataLoader(Dataset.from_dict({"text": ["query", "query"]}), batch_size=2),
            DataLoader(
                Dataset.from_dict({"text": ["short", "longer document"]}), batch_size=2
            ),
            task_metadata=MockRetrievalTask.metadata,
            hf_split="test",
            hf_subset="default",
            **options,
        )
    assert options == original_options
    assert calls == ([2] if adapter == "sentence_transformers" else [1, 1])
    np.testing.assert_array_equal(result, expected)
    if adapter == "sentence_transformers":
        assert result is expected
    if precision in {None, "float32"}:
        assert not caught
    else:
        assert len(caught) == 1
        assert "Ignoring precision=" in str(caught[0].message)
