from __future__ import annotations

import typing

import pytest
from datasets import Dataset
from torch.utils.data import DataLoader

from mteb.abstasks.task_metadata import SimplifiedTaskType, TaskMetadata
from mteb.models.model_implementations.jina_models import (
    _SIMPLIFIED_TO_JINA_TASK,
    JinaV5OmniWrapper,
    jina_embeddings_v5_omni_small,
)
from mteb.types import PromptType
from tests.mock_models import MockSentenceTransformer
from tests.mock_tasks import MockRetrievalTask

_VARIANT_MAP = {
    "Retrieval": "retrieval",
    "Clustering": "clustering",
    "STS": "text-matching",
    "PairClassification": "text-matching",
    "Classification": "classification",
}


class CapturingModel(MockSentenceTransformer):
    """Records encode kwargs for assertion."""

    def __init__(self):
        super().__init__()
        self.captured: list[dict] = []

    def encode(self, inputs, **kwargs):
        self.captured.append(
            {
                "inputs": inputs,
                "task": kwargs.get("task"),
                "prompt": kwargs.get("prompt"),
            }
        )
        return super().encode(inputs, **kwargs)


def _make_wrapper(model_prompts=_VARIANT_MAP) -> JinaV5OmniWrapper:
    return JinaV5OmniWrapper(model=CapturingModel(), model_prompts=model_prompts)


def _task(task_type: str) -> TaskMetadata:
    meta = MockRetrievalTask.metadata.model_copy()
    meta.type = task_type
    return meta


def _encode(wrapper: JinaV5OmniWrapper, prompt_type: PromptType, task_type: str):
    wrapper.encode(
        DataLoader(Dataset.from_dict({"text": ["text"]})),
        task_metadata=_task(task_type),
        hf_split="test",
        hf_subset="default",
        prompt_type=prompt_type,
    )


def _omni_prompts() -> dict:
    return jina_embeddings_v5_omni_small.loader_kwargs["model_prompts"]


@pytest.mark.parametrize(
    "prompt_type,task_type,expected_task,expected_prompt",
    [
        (PromptType.query, "Retrieval", "retrieval", "Query: "),
        (PromptType.document, "Retrieval", "retrieval", "Document: "),
        (PromptType.document, "Clustering", "clustering", ""),
        (PromptType.query, "STS", "text-matching", ""),
        (PromptType.document, "Classification", "classification", ""),
        (PromptType.query, "PairClassification", "text-matching", ""),
    ],
)
def test_prefix_by_task_type(prompt_type, task_type, expected_task, expected_prompt):
    wrapper = _make_wrapper()
    _encode(wrapper, prompt_type, task_type)
    assert wrapper.model.captured[-1]["task"] == expected_task
    assert wrapper.model.captured[-1]["prompt"] == expected_prompt


@pytest.mark.parametrize(
    "prompt_type,task_type,expected_task,expected_prompt",
    [
        (PromptType.query, "Any2AnyRetrieval", "retrieval", "Query: "),
        (PromptType.document, "ImageClustering", "clustering", ""),
        (PromptType.query, "VisualSTS(eng)", "text-matching", ""),
        (PromptType.query, "AudioPairClassification", "text-matching", ""),
    ],
)
def test_simplified_fallback(prompt_type, task_type, expected_task, expected_prompt):
    """Simplified task types fall back to the _SIMPLIFIED_TO_JINA_TASK map."""
    wrapper = _make_wrapper(model_prompts={})
    _encode(wrapper, prompt_type, task_type)
    assert wrapper.model.captured[-1]["task"] == expected_task
    assert wrapper.model.captured[-1]["prompt"] == expected_prompt


@pytest.mark.parametrize(
    "prompt_type,task_type,expected_task,expected_prompt",
    [
        (PromptType.query, "ImageClassification", "retrieval", "Query: "),
        (PromptType.query, "AudioClassification", "retrieval", "Query: "),
        (PromptType.query, "ZeroShotClassification", "retrieval", "Query: "),
        (PromptType.document, "Compositionality", "clustering", ""),
        (PromptType.query, "AudioPairClassification", "retrieval", "Query: "),
        (PromptType.document, "ImageClustering", "text-matching", ""),
    ],
)
def test_omni_overrides(prompt_type, task_type, expected_task, expected_prompt):
    """Omni model_prompts overrides route specific task types correctly."""
    wrapper = _make_wrapper(model_prompts=_omni_prompts())
    _encode(wrapper, prompt_type, task_type)
    assert wrapper.model.captured[-1]["task"] == expected_task
    assert wrapper.model.captured[-1]["prompt"] == expected_prompt


def test_simplified_to_jina_task_covers_all_simplified_types():
    """Every SimplifiedTaskType must have a Jina LoRA mapping."""
    expected = set(typing.get_args(SimplifiedTaskType))
    missing = expected - set(_SIMPLIFIED_TO_JINA_TASK.keys())
    assert not missing, (
        f"Missing entries in _SIMPLIFIED_TO_JINA_TASK: {missing}. "
        "Add them to keep routing in sync with mteb SimplifiedTaskType."
    )
