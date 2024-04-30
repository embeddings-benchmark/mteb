from __future__ import annotations

import pytest

import mteb
from mteb import get_tasks
from mteb.abstasks.TaskMetadata import TASK_DOMAIN, TASK_TYPE
from mteb.overview import MTEBTasks


def test_get_tasks_size_differences():
    assert len(get_tasks()) > 0
    assert len(get_tasks()) >= len(get_tasks(languages=["eng"]))
    assert len(get_tasks()) >= len(get_tasks(script=["Latn"]))
    assert len(get_tasks()) >= len(get_tasks(domains=["Legal"]))
    assert len(get_tasks()) >= len(get_tasks(languages=["eng", "deu"]))
    assert len(get_tasks(languages=["eng", "deu"])) >= len(
        get_tasks(languages=["eng", "deu"])
    )


@pytest.mark.parametrize("languages", [["eng", "deu"], ["eng"], None])
@pytest.mark.parametrize("script", [["Latn"], ["Cyrl"], None])
@pytest.mark.parametrize("domains", [["Legal"], ["Medical", "Non-fiction"], None])
@pytest.mark.parametrize("task_types", [["Classification"], ["Clustering"], None])
@pytest.mark.parametrize("exclude_superseeded_datasets", [True, False])
def test_get_task(
    languages: list[str],
    script: list[str],
    domains: list[TASK_DOMAIN],
    task_types: list[TASK_TYPE] | None,
    exclude_superseeded_datasets: bool,
):
    tasks = mteb.get_tasks(
        languages=languages,
        script=script,
        domains=domains,
        task_types=task_types,
        exclude_superseeded=exclude_superseeded_datasets,
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
        if exclude_superseeded_datasets:
            assert task.superseeded_by is None


def test_get_tasks_filtering():
    """Tests that get_tasks filters tasks for languages within the task, i.e. that a multilingual task returns only relevant subtasks for the
    specified languages
    """
    tasks = get_tasks(languages=["eng"])

    for task in tasks:
        if task.is_multilingual:
            assert isinstance(task.metadata.eval_langs, dict)

            for hf_split in task.langs:
                assert "eng-Latn" in task.metadata.eval_langs[hf_split]


@pytest.mark.parametrize("script", [["Latn"], ["Cyrl"], None])
@pytest.mark.parametrize("task_types", [["Classification"], ["Clustering"], None])
def test_MTEBTasks(
    script: list[str],
    task_types: list[TASK_TYPE] | None,
):
    tasks = mteb.get_tasks(script=script, task_types=task_types)
    assert isinstance(tasks, MTEBTasks)
    langs = tasks.languages
    for t in tasks:
        assert len(langs.intersection(t.languages)) > 0

    # check for header of a table
    n_langs = len(tasks)
    assert len(tasks.to_markdown().split("\n")) - 3 == n_langs
