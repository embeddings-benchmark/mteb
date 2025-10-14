import logging

import pytest

from mteb.abstasks import AbsTask
from mteb.models.abs_encoder import AbsEncoder
from tests.task_grid import TASK_TEST_GRID

logging.basicConfig(level=logging.INFO)


@pytest.mark.parametrize(
    "task_name",
    TASK_TEST_GRID
    + (
        "BitextMining",
        "Classification",
        "MultilabelClassification",
        "Clustering",
        "PairClassification",
        "Reranking",
        "Retrieval",
        "STS",
        "Summarization",
        "InstructionRetrieval",
        "InstructionReranking",
    ),
)
def test_validate_task_to_prompt_name(task_name: str | AbsTask):
    if isinstance(task_name, AbsTask):
        task_names = [task_name.metadata.name]
    else:
        task_names = [task_name]

    model_prompts = dict.fromkeys(task_names, "prompt_name")
    model_prompts |= {task_name + "-query": "prompt_name" for task_name in task_names}
    model_prompts |= {
        task_name + "-document": "prompt_name" for task_name in task_names
    }
    model_prompts |= {
        "query": "prompt_name",
        "document": "prompt_name",
    }
    AbsEncoder.validate_task_to_prompt_name(model_prompts)


@pytest.mark.parametrize("raise_for_invalid_keys", (True, False))
def test_validate_task_to_prompt_name_for_none(raise_for_invalid_keys: bool):
    result = AbsEncoder.validate_task_to_prompt_name(
        None, raise_for_invalid_keys=raise_for_invalid_keys
    )
    assert result is None if raise_for_invalid_keys else (None, None)


@pytest.mark.parametrize(
    "task_prompt_dict",
    [
        {"task_name": "prompt_name"},
        {"task_name-query": "prompt_name"},
        {"task_name-task_name": "prompt_name"},
    ],
)
def test_validate_task_to_prompt_name_fails_and_raises(
    task_prompt_dict: dict[str, str],
):
    with pytest.raises(KeyError):
        AbsEncoder.validate_task_to_prompt_name(task_prompt_dict)


@pytest.mark.parametrize(
    "task_prompt_dict, expected_valid, expected_invalid",
    [
        ({"task_name": "prompt_name"}, 0, 1),
        ({"task_name-query": "prompt_name"}, 0, 1),
        (
            {
                "task_name-query": "prompt_name",
                "query": "prompt_name",
                "Retrieval": "prompt_name",
            },
            2,
            1,
        ),
        ({"task_name-task_name": "prompt_name"}, 0, 1),
    ],
)
def test_validate_task_to_prompt_name_filters_and_reports(
    task_prompt_dict: dict[str, str], expected_valid: int, expected_invalid: int
):
    valid, invalid = AbsEncoder.validate_task_to_prompt_name(
        task_prompt_dict, raise_for_invalid_keys=False
    )
    assert len(valid) == expected_valid
    assert len(invalid) == expected_invalid
