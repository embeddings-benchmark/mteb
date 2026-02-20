"""Test if the metadata of all tasks is filled and valid."""

import pytest

from mteb.abstasks import AbsTask
from mteb.get_tasks import get_tasks

# Historic datasets without filled metadata. Do NOT add new datasets to this list.
# Tasks should be removed from this list once their metadata is filled.
_HISTORIC_DATASETS = []


@pytest.mark.parametrize(
    "task", get_tasks(exclude_superseded=False, exclude_aggregate=False)
)
def test_all_metadata_is_filled_and_valid(task: AbsTask):
    # --- test metadata is filled and valid ---
    if task.metadata.name not in _HISTORIC_DATASETS:
        task.metadata._validate_metadata()
        assert task.metadata.is_filled(), (
            f"Metadata for {task.metadata.name} is not filled"
        )
    else:
        assert not task.metadata.is_filled(), (
            f"Metadata for {task.metadata.name} is stated as not filled (historic), but it is filled, please remove the dataset from the historic list."
        )

    # --- Check that no dataset trusts remote code ---
    assert task.metadata.dataset.get("trust_remote_code", False) is False, (
        f"Dataset {task.metadata.name} should not trust remote code"
    )

    # --- Test is descriptive stats are present for all datasets ---
    if task.is_aggregate:  # aggregate tasks do not have descriptive stats
        return

    # TODO https://github.com/embeddings-benchmark/mteb/issues/3498
    if task.metadata.name in (
        "FleursA2TRetrieval",
        "FleursT2ARetrieval",
        "SoundDescsA2TRetrieval",
        "SoundDescsT2ARetrieval",
        "BirdSet",
        "AudioSet",
    ):
        assert task.metadata.descriptive_stats is None
        pytest.skip("Skipping audio tasks for now, see issue #3498")

    assert task.metadata.descriptive_stats is not None, (
        f"Dataset {task.metadata.name} should have descriptive stats. You can add metadata to your task by running `YourTask().calculate_descriptive_statistics()`"
    )
    assert task.metadata.n_samples is not None
