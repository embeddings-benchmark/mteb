import pytest

from mteb import get_tasks
from mteb.abstasks.abstask import AbsTask
from mteb.abstasks.task_metadata import TaskDomain, TaskType
from mteb.filter_tasks import filter_tasks
from mteb.tasks.aggregated_tasks import CQADupstackRetrieval
from mteb.types import Modalities


@pytest.fixture
def all_tasks():
    return get_tasks()


def test_get_tasks_size_differences(all_tasks: list[AbsTask]):
    assert len(all_tasks) > 0
    assert len(all_tasks) >= len(filter_tasks(all_tasks, script=["Latn"]))
    assert len(all_tasks) >= len(filter_tasks(all_tasks, domains=["Legal"]))
    assert len(all_tasks) >= len(filter_tasks(all_tasks, languages=["eng", "deu"]))
    text_task = filter_tasks(all_tasks, modalities=["text"])
    assert len(all_tasks) >= len(text_task)
    assert len(filter_tasks(all_tasks, modalities=["text", "image"])) >= len(text_task)


@pytest.mark.parametrize("languages", [["eng", "deu"], ["eng"], None])
@pytest.mark.parametrize("script", [["Latn"], ["Cyrl"], None])
@pytest.mark.parametrize("domains", [["Legal"], ["Medical", "Non-fiction"], None])
@pytest.mark.parametrize("task_types", [["Classification"], None])
def test_filter_tasks(
    all_tasks: list[AbsTask],
    languages: list[str],
    script: list[str],
    domains: list[TaskDomain],
    task_types: list[TaskType] | None,  # type: ignore
):
    """Tests that get_tasks filters tasks correctly. This could in principle be combined with the following tests, but they have been kept
    separate to reduce the grid size.
    """
    tasks = filter_tasks(
        all_tasks,
        languages=languages,
        script=script,
        domains=domains,
        task_types=task_types,
    )

    for task in tasks:
        if languages:
            assert set(languages).intersection(task.metadata.languages)
        if script:
            assert set(script).intersection(task.metadata.scripts)
        if domains:
            task_domains = (
                set(task.metadata.domains) if task.metadata.domains else set()
            )
            assert set(domains).intersection(set(task_domains))
        if task_types:
            assert task.metadata.type in task_types


@pytest.mark.parametrize("languages", [["eng", "deu"], ["eng"]])
@pytest.mark.parametrize("domains", [["Medical", "Non-fiction"], None])
@pytest.mark.parametrize("task_types", [["Classification"], None])
@pytest.mark.parametrize("exclude_superseded_datasets", [True, False])
def test_filter_tasks_superseded(
    all_tasks: list[AbsTask],
    languages: list[str],
    domains: list[TaskDomain],
    task_types: list[TaskType] | None,  # type: ignore
    exclude_superseded_datasets: bool,
):
    tasks = filter_tasks(
        all_tasks,
        languages=languages,
        domains=domains,
        task_types=task_types,
        exclude_superseded=exclude_superseded_datasets,
    )

    for task in tasks:
        if languages:
            assert set(languages).intersection(task.metadata.languages)
        if domains:
            task_domains = (
                set(task.metadata.domains) if task.metadata.domains else set()
            )
            assert set(domains).intersection(set(task_domains))
        if task_types:
            assert task.metadata.type in task_types
        if exclude_superseded_datasets:
            assert task.superseded_by is None


@pytest.mark.parametrize("languages", [["eng", "deu"], ["eng"]])
@pytest.mark.parametrize("modalities", [["text"], ["image"], ["text", "image"], None])
@pytest.mark.parametrize("exclusive_modality_filter", [True, False])
def test_filter_tasks_modalities(
    all_tasks: list[AbsTask],
    languages: list[str],
    modalities: list[Modalities] | None,
    exclusive_modality_filter: bool,
):
    tasks = filter_tasks(
        all_tasks,
        languages=languages,
        modalities=modalities,
        exclusive_modality_filter=exclusive_modality_filter,
    )

    for task in tasks:
        if languages:
            assert set(languages).intersection(task.metadata.languages)
        if modalities:
            if exclusive_modality_filter:
                assert set(task.modalities) == set(modalities)
            else:
                assert any(mod in task.modalities for mod in modalities)


@pytest.mark.parametrize("languages", [["eng", "deu"], ["eng"], None])
@pytest.mark.parametrize("script", [["Latn"], ["Cyrl"], None])
@pytest.mark.parametrize("exclude_aggregate", [True, False])
def test_filter_tasks_exclude_aggregate(
    all_tasks: list[AbsTask],
    languages: list[str],
    script: list[str],
    exclude_aggregate: bool,
):
    tasks = filter_tasks(
        all_tasks,
        languages=languages,
        script=script,
        exclude_aggregate=exclude_aggregate,
    )

    for task in tasks:
        if languages:
            assert set(languages).intersection(task.metadata.languages)
        if script:
            assert set(script).intersection(task.metadata.scripts)
        if exclude_aggregate:
            # Aggregate tasks should be excluded
            assert not task.is_aggregate


def test_filter_tasks_exclude_aggregate_with_task_classes():
    """Test that filter_tasks correctly handles aggregate tasks when passed task classes.

    Regression test for PR #3372
    """
    tasks = filter_tasks([CQADupstackRetrieval], exclude_aggregate=True)
    assert len(tasks) == 0

    tasks = filter_tasks([CQADupstackRetrieval], exclude_aggregate=False)
    assert len(tasks) == 1
