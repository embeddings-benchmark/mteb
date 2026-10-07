from typing import Any

import numpy as np
import pytest
from datasets import Dataset

from mteb.mocks.mock_tasks import MockRetrievalTask
from mteb.models.search_wrappers import SearchCrossEncoderWrapper


class MockCrossEncoder:
    mteb_model_meta = None

    def predict(self, inputs1, inputs2, **kwargs: Any):
        self.kwargs = kwargs
        self.loader_batch_sizes = (inputs1.batch_size, inputs2.batch_size)
        return np.array([0.75, 0.25])


@pytest.mark.parametrize("batch_size", [None, 1, 3])
@pytest.mark.parametrize("show_progress_bar", [None, False, True])
def test_search_forwards_inference_options(batch_size, show_progress_bar):
    model = MockCrossEncoder()
    wrapper = SearchCrossEncoderWrapper(model)
    context = dict(
        task_metadata=MockRetrievalTask.metadata,
        hf_split="test",
        hf_subset="default",
    )
    inference_options = {}
    if batch_size is not None:
        inference_options["batch_size"] = batch_size
    if show_progress_bar is not None:
        inference_options["show_progress_bar"] = show_progress_bar
    options = {**inference_options, "precision": "float32"}
    wrapper.index(
        Dataset.from_dict({"id": ["d1", "d2"], "text": ["relevant", "other"]}),
        **context,
        encode_kwargs=options,
    )
    results = wrapper.search(
        Dataset.from_dict({"id": ["q1"], "text": ["question"]}),
        **context,
        top_k=2,
        top_ranked={"q1": ["d1", "d2"]},
        encode_kwargs=options,
    )
    assert model.kwargs == {**context, **inference_options}
    assert model.loader_batch_sizes == (batch_size or 32, batch_size or 32)
    assert options == {**inference_options, "precision": "float32"}
    assert results == {"q1": {"d1": 0.75, "d2": 0.25}}


@pytest.mark.parametrize("show_progress_bar", [None, False, True])
def test_kalm_progress_control(show_progress_bar, monkeypatch):
    from torch.utils.data import DataLoader

    from mteb.models.model_implementations import kalm_reranker

    model = kalm_reranker.KaLMRerankerWrapper.__new__(kalm_reranker.KaLMRerankerWrapper)
    model.instruction = "instruction"
    progress = []

    def tqdm(iterable, *, disable, desc):
        progress.append(disable)
        return iterable

    monkeypatch.setattr(kalm_reranker, "tqdm", tqdm)
    monkeypatch.setattr(
        model, "_predict_batch", lambda pairs, instructions: [0.75] * len(pairs)
    )
    options = (
        {} if show_progress_bar is None else {"show_progress_bar": show_progress_bar}
    )
    loader = DataLoader(Dataset.from_dict({"text": ["example"]}), batch_size=1)
    result = model.predict(
        loader,
        loader,
        task_metadata=MockRetrievalTask.metadata,
        hf_split="test",
        hf_subset="default",
        **options,
    )
    assert progress == [show_progress_bar is not True]
    np.testing.assert_array_equal(result, [0.75])
