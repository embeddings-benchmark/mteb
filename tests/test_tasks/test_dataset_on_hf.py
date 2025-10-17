import logging
import os

import huggingface_hub
import pytest

import mteb
from mteb.abstasks.aggregated_task import AbsTaskAggregate
from mteb.get_tasks import get_tasks
from tests.task_grid import (
    MOCK_MIEB_TASK_GRID_AS_STRING,
    MOCK_TASK_TEST_GRID_AS_STRING,
)

logging.basicConfig(level=logging.INFO)

ALL_MOCK_TASKS = MOCK_TASK_TEST_GRID_AS_STRING + MOCK_MIEB_TASK_GRID_AS_STRING

tasks = [
    t
    for t in get_tasks(exclude_superseded=False)
    if t.metadata.name not in ALL_MOCK_TASKS
]


datasets_not_available = [
    "AfriSentiLangClassification",
]


_original_dataset_revisions = list(
    {  # deduplicate as multiple tasks rely on the same dataset (save us at least 100 test cases)
        (t.metadata.dataset["path"], t.metadata.dataset["revision"])
        for t in mteb.get_tasks(exclude_superseded=False)
        if not isinstance(t, AbsTaskAggregate)
        and t.metadata.name not in datasets_not_available
        and t.metadata.name not in ALL_MOCK_TASKS
    }
)

custom_revisions = os.getenv("CUSTOM_DATASET_REVISIONS")
if custom_revisions:
    # Parse comma-separated list of "path:revision" pairs
    dataset_revisions = [
        tuple(pair.split(":", 1)) for pair in custom_revisions.split(",") if ":" in pair
    ]
else:
    dataset_revisions = _original_dataset_revisions


@pytest.mark.test_datasets
@pytest.mark.flaky(
    reruns=5,
    reruns_delay=12,
    only_rerun=["AssertionError"],
    reason="May fail due to network issues",
)
@pytest.mark.parametrize("dataset_revision", dataset_revisions)
def test_dataset_on_hf(dataset_revision: tuple[str, str]):
    repo_id, revision = dataset_revision
    try:
        huggingface_hub.dataset_info(repo_id, revision=revision)
    except (
        huggingface_hub.errors.RepositoryNotFoundError,
        huggingface_hub.errors.RevisionNotFoundError,
    ):
        assert False, f"Dataset {repo_id} - {revision} not available"
    except Exception as e:
        assert False, f"Dataset {repo_id} - {revision} failed with {e}"
