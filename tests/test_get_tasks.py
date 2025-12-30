import pytest

import mteb
from mteb import get_task, get_tasks
from mteb.abstasks.abstask import AbsTask
from mteb.abstasks.task_metadata import TaskType
from mteb.get_tasks import MTEBTasks
from mteb.types import Modalities


@pytest.fixture
def all_tasks():
    return get_tasks()


@pytest.mark.parametrize(
    "task_name", ["BornholmBitextMining", "CQADupstackRetrieval", "Birdsnap"]
)
@pytest.mark.parametrize("eval_splits", [["test"], None])
def test_get_task(
    task_name: str,
    eval_splits: list[str] | None,
):
    task = get_task(
        task_name,
        eval_splits=eval_splits,
    )
    assert isinstance(task, AbsTask)
    assert task.metadata.name == task_name
    if eval_splits:
        for split in task.eval_splits:
            assert split in eval_splits
    else:
        assert task.eval_splits == task.metadata.eval_splits


def test_get_tasks_filtering():
    """Tests that get_tasks filters tasks for languages within the task, i.e. that a multilingual task returns only relevant subtasks for the
    specified languages
    """
    tasks = get_tasks(languages=["eng"])

    for task in tasks:
        if (
            task.metadata.is_multilingual
            and task.metadata.name != "STS17MultilingualVisualSTSEng"
        ):
            assert isinstance(task.metadata.eval_langs, dict)

            for hf_subset in task.hf_subsets:
                assert "eng-Latn" in task.metadata.eval_langs[hf_subset], (
                    f"{task.metadata.name}"
                )


@pytest.mark.parametrize("script", [["Cyrl"], None])
@pytest.mark.parametrize("task_types", [["Classification"], ["Clustering"], None])
@pytest.mark.parametrize("modalities", [["text"], ["image"], None])
def test_mteb_mteb_tasks(
    script: list[str],
    task_types: list[TaskType] | None,
    modalities: list[Modalities] | None,
):
    tasks = mteb.get_tasks(script=script, task_types=task_types, modalities=modalities)
    assert isinstance(tasks, MTEBTasks)
    langs = tasks.languages
    for t in tasks:
        assert len(langs.intersection(t.languages)) > 0

    # check for header of a table
    n_langs = len(tasks)
    assert len(tasks.to_markdown().split("\n")) - 3 == n_langs


@pytest.mark.parametrize("modalities", [["text"], ["image"], ["text", "image"]])
def test_get_tasks_with_exclusive_modality_filter(modalities):
    """Test exclusive_modality_filter with actual tasks (if available)"""
    text_tasks_exclusive = get_tasks(
        modalities=modalities, exclusive_modality_filter=True
    )
    for task in text_tasks_exclusive:
        assert set(task.modalities) == set(modalities)


def test_get_tasks_privacy_filtering():
    """Test that get_tasks correctly filters by privacy status"""
    # By default, should only return public datasets (exclude_private=True)
    public_tasks = get_tasks()

    # Should include private datasets when explicitly requested
    all_tasks = get_tasks(exclude_private=False)

    # All tasks should contain at least as many or more tasks than public tasks
    assert len(all_tasks) >= len(public_tasks)

    # All returned tasks should be public when exclude_private=True
    for task in public_tasks:
        assert (
            task.metadata.is_public is not False
        )  # None or True are both considered public
