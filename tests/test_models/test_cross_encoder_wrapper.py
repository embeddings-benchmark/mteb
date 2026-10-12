from typing import Any

import numpy as np
from datasets import Dataset
from sentence_transformers import CrossEncoder

from mteb._create_dataloaders import create_dataloader
from mteb.mocks.mock_tasks import MockRetrievalTask
from mteb.models.sentence_transformer_wrapper import CrossEncoderWrapper


def test_cross_encoder_predicts_one_batch_at_a_time():
    """Pairs are scored per dataloader batch (not all at once) with unchanged scores."""
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
    queries = ["what is a cat", "what is a dog", "capital of france", "red", "sky"]
    documents = ["a small feline", "a loyal pet", "paris", "a colour", "blue"]
    expected = cross_encoder.predict(list(zip(queries, documents, strict=True)))

    predict = cross_encoder.predict
    batch_sizes = []

    def recording_predict(pairs: list[tuple[str, str]], **kwargs: Any) -> np.ndarray:
        batch_sizes.append(len(pairs))
        return predict(pairs, **kwargs)

    cross_encoder.predict = recording_predict
    model = CrossEncoderWrapper(cross_encoder)
    task_metadata = MockRetrievalTask.metadata
    scores = model.predict(
        create_dataloader(
            Dataset.from_dict({"text": queries}),
            task_metadata=task_metadata,
            batch_size=2,
        ),
        create_dataloader(
            Dataset.from_dict({"text": documents}),
            task_metadata=task_metadata,
            batch_size=2,
        ),
        task_metadata=task_metadata,
        hf_split="test",
        hf_subset="default",
    )

    assert batch_sizes == [2, 2, 1]
    np.testing.assert_allclose(scores, expected, rtol=1e-5)
