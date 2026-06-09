from __future__ import annotations

from typing import Any

import pytest
from torch.utils.data import DataLoader

import mteb
from mteb.abstasks.task_metadata import TaskMetadata
from mteb.models import ModelMeta
from mteb.models.model_implementations.random_baseline import (
    RandomEncoderBaseline,
)
from mteb.models.model_meta import ScoringFunction
from mteb.types import PromptType
from mteb.types._encoder_io import Array, BatchedInput
from tests.mock_tasks import (
    MockAudioClassification,
    MockClassificationTask,
    MockImageClassificationTask,
    MockInstructionRetrieval,
    MockRetrievalDialogTask,
    MockRetrievalTask,
    MockVideoAudioClassification,
    MockVideoAudioRetrievalT2VA,
    MockVideoAudioTextRetrievalVAT2T,
    MockVideoClassification,
)

_MODALITY_COLUMNS = frozenset({"text", "image", "audio", "video"})


class TrackingEncoderModel(RandomEncoderBaseline):
    mteb_model_meta = ModelMeta.create_empty(
        overwrites=dict(
            name="mock/model",
            revision="1",
            modalities=["text", "image", "audio", "video"],
            similarity_fn_name=ScoringFunction.COSINE,
        )
    )

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            model_name=self.mteb_model_meta.name,
            revision=self.mteb_model_meta.revision,
            **kwargs,
        )
        self.observed: dict[PromptType | None, set[str]] = {}

    def encode(
        self,
        inputs: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> Array:
        first_batch = next(iter(inputs))
        self.observed.setdefault(prompt_type, set()).update(first_batch.keys())
        return super().encode(
            inputs,
            task_metadata=task_metadata,
            hf_split=hf_split,
            hf_subset=hf_subset,
            prompt_type=prompt_type,
            **kwargs,
        )


@pytest.mark.parametrize(
    "task",
    [
        MockClassificationTask(),
        MockImageClassificationTask(),
        MockAudioClassification(),
        MockVideoClassification(),
        MockVideoAudioClassification(),
        MockVideoAudioRetrievalT2VA(),
        MockVideoAudioTextRetrievalVAT2T(),
    ],
    ids=lambda task: task.metadata.name,
)
def test_modality_columns_passed_to_encode(task):
    if "image" in task.metadata.modalities:
        pytest.importorskip(
            "torchvision", reason="Image dependencies are not installed"
        )
    if "audio" in task.metadata.modalities:
        pytest.importorskip("torchaudio", reason="Audio dependencies are not installed")
    if "video" in task.metadata.modalities:
        pytest.importorskip("torchcodec", reason="Video dependencies are not installed")

    model = TrackingEncoderModel()
    mteb.evaluate(model, task, cache=None)

    assert model.observed, "encode() was never called"

    for prompt_type, observed_columns in model.observed.items():
        expected_modalities = set(task.metadata.get_modalities(prompt_type))
        observed_modality_columns = observed_columns & _MODALITY_COLUMNS

        missing = expected_modalities - observed_modality_columns
        assert not missing, (
            f"prompt_type={prompt_type!r}: expected modality columns "
            f"{expected_modalities!r} but {missing!r} were never seen. "
            f"Got: {observed_columns!r}"
        )

        unexpected = observed_modality_columns - expected_modalities
        assert not unexpected, (
            f"prompt_type={prompt_type!r}: unexpected modality columns "
            f"{unexpected!r} were passed to encode. "
            f"Expected only: {expected_modalities!r}"
        )


@pytest.mark.parametrize(
    "task, expected_columns_by_prompt_type",
    [
        pytest.param(
            MockRetrievalTask(),
            {
                PromptType.query: {"text", "query"},
                PromptType.document: {"text", "body", "title"},
            },
            id=MockRetrievalTask.metadata.name,
        ),
        pytest.param(
            MockInstructionRetrieval(),
            {
                PromptType.query: {"text", "query", "instruction"},
                PromptType.document: {"text", "body", "title"},
            },
            id=MockInstructionRetrieval.metadata.name,
        ),
        pytest.param(
            MockRetrievalDialogTask(),
            {
                PromptType.query: {"text", "query", "conversation"},
                PromptType.document: {"text", "body", "title"},
            },
            id=MockRetrievalDialogTask.metadata.name,
        ),
    ],
)
def test_text_retrieval_columns_passed_to_encode(task, expected_columns_by_prompt_type):
    model = TrackingEncoderModel()
    mteb.evaluate(model, task, cache=None)

    assert model.observed, "encode() was never called"

    for prompt_type, expected_columns in expected_columns_by_prompt_type.items():
        assert prompt_type in model.observed, (
            f"encode() was never called with prompt_type={prompt_type!r}"
        )
        observed_columns = model.observed[prompt_type]

        missing = expected_columns - observed_columns
        assert not missing, (
            f"prompt_type={prompt_type!r}: expected columns {expected_columns!r} "
            f"but {missing!r} were never seen. Got: {observed_columns!r}"
        )
