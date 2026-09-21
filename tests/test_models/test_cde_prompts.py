from typing import Any

import numpy as np
import pytest

from mteb.abstasks.task_metadata import TaskMetadata
from mteb.mocks.mock_tasks import (
    LegacyMockClusteringFastTask,
    MockClassificationTask,
    MockRetrievalTask,
)
from mteb.models.model_implementations.cde_models import (
    CDEWrapper,
    cde_model_prompts,
)
from mteb.types import PromptType


class _RecordingModel:
    def __init__(self) -> None:
        self.prompts: list[str | None] = []

    def encode(self, sentences: list[str], prompt: str | None = None, **kwargs: Any):
        self.prompts.append(prompt)
        return np.zeros((len(sentences), 2))


class _CDEWithoutDownload(CDEWrapper):
    def __init__(self) -> None:
        self.model = _RecordingModel()
        self.model_prompts = cde_model_prompts
        self.dataset_embeddings = np.zeros((1, 2))


@pytest.mark.parametrize(
    ("task_metadata", "prompt_type", "expected"),
    [
        (MockRetrievalTask.metadata, PromptType.query, "search_query: "),
        (MockRetrievalTask.metadata, PromptType.document, "search_document: "),
        (MockClassificationTask.metadata, None, "classification: "),
        (LegacyMockClusteringFastTask.metadata, None, "clustering: "),
    ],
)
def test_cde_encode_passes_prompt_text_to_model(
    task_metadata: TaskMetadata, prompt_type: PromptType | None, expected: str
):
    model = _CDEWithoutDownload()
    # dataset embeddings for this task/subset are already loaded
    model.prev_embeddings_key = model._create_embeddings_key(task_metadata, "default")

    model.encode(
        [{"text": ["a sentence"]}],  # type: ignore[arg-type]
        task_metadata=task_metadata,
        hf_split="test",
        hf_subset="default",
        prompt_type=prompt_type,
    )

    assert model.model.prompts == [expected]
