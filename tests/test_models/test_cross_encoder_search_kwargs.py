from typing import Any

import mteb
from mteb.mocks.mock_tasks import MockRerankingTask
from mteb.types import Array


def test_evaluate_forwards_encode_kwargs_to_cross_encoder():
    """Test that the encode_kwargs are forwarded to the cross-encoder's predict method."""
    model = mteb.get_model("mteb/baseline-random-cross-encoder")
    predict = model.predict
    received = {}

    def recording_predict(*args: Any, **kwargs: Any) -> Array:
        received.update(kwargs)
        return predict(*args, **kwargs)

    model.predict = recording_predict
    encode_kwargs = {"batch_size": 3, "show_progress_bar": False}
    mteb.evaluate(model, MockRerankingTask(), encode_kwargs=encode_kwargs, cache=None)

    assert received.items() >= encode_kwargs.items()
