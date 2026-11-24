from __future__ import annotations

import numpy as np
import pytest
from datasets import Dataset
from torch.utils.data import DataLoader

from mteb.mocks.mock_tasks import MockRetrievalTask
from mteb.models.model_implementations.nomic_models import NomicWrapper, model_prompts
from tests.mock_models import MockSentenceTransformer


@pytest.mark.parametrize(
    ("task_type", "expected_norm"),
    [
        # layer norm only, as in nomic's own evaluation of classification
        ("Classification", np.sqrt(10)),
        ("Clustering", 1.0),
        ("Retrieval", 1.0),
    ],
)
def test_nomic_v1_5_normalizes_embeddings_by_task_type(
    task_type: str, expected_norm: float
):
    wrapper = NomicWrapper(
        MockSentenceTransformer(), revision=None, model_prompts=model_prompts
    )
    wrapper.model_name = "nomic-ai/nomic-embed-text-v1.5"
    task_metadata = MockRetrievalTask.metadata.model_copy()
    task_metadata.type = task_type

    embeddings = wrapper.encode(
        DataLoader(Dataset.from_dict({"text": ["first text", "second text"]})),
        task_metadata=task_metadata,
        hf_split="test",
        hf_subset="default",
    )

    np.testing.assert_allclose(
        np.linalg.norm(embeddings, axis=1), expected_norm, rtol=1e-4
    )
