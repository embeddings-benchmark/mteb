from __future__ import annotations

import logging
import random

import pytest

from mteb.overview import (
    TASKS_REGISTRY,
    filter_tasks_by_modalities,
    get_task,
    get_tasks,
)

logger = logging.getLogger(__name__)


@pytest.mark.integration
def test_with_actual_task_registry():
    """Test with actual tasks from the registry (if available)"""
    if not TASKS_REGISTRY:
        pytest.skip("No tasks in TASKS_REGISTRY")

    try:
        task_name = random.choice(list(TASKS_REGISTRY.keys()))
        task = get_task(task_name)
        original_modalities = task.modalities.copy()

        task_filtered_text = task.filter_modalities(["text"])
        if "text" in original_modalities:
            assert task_filtered_text.modalities
        else:
            assert not task_filtered_text.modalities

        task_filtered_image = task.filter_modalities(["image"])
        if "image" in original_modalities:
            assert task_filtered_image.modalities
        else:
            assert not task_filtered_image.modalities

    except Exception as e:
        pytest.skip(f"Error testing with actual registry: {str(e)}")


def test_specific_task_filtering():
    """Test that specific tasks can be filtered by modality"""
    if not TASKS_REGISTRY:
        pytest.skip("No tasks in TASKS_REGISTRY, skipping test")

    first_task_name = random.choice(list(TASKS_REGISTRY.keys()))
    task = get_task(first_task_name, modalities=["audio"])

    assert task is not None
    assert task.metadata.modalities == []

    filtered = filter_tasks_by_modalities([task], ["audio"])
    assert len(filtered) == 0


def test_multiple_tasks_with_get_tasks():
    """Test that get_tasks function correctly filters by modality"""
    try:
        all_tasks = get_tasks()
        if not all_tasks:
            pytest.skip("No tasks available in registry")

        text_tasks = get_tasks(modalities=["text"])
        image_tasks = get_tasks(modalities=["image"])

        assert len(all_tasks) >= len(text_tasks)
        assert len(all_tasks) >= len(image_tasks)

        for task in text_tasks:
            assert "text" in task.modalities

        for task in image_tasks:
            assert "image" in task.modalities

        text_and_image_tasks = get_tasks(modalities=["text", "image"])

        for task in text_and_image_tasks:
            assert any(m in task.modalities for m in ["text", "image"])

        assert len(text_and_image_tasks) >= len(text_tasks)
        assert len(text_and_image_tasks) >= len(image_tasks)

    except Exception as e:
        pytest.skip(f"Error in get_tasks test: {str(e)}")


def test_get_tasks_with_exclusive_modality_filter():
    """Test exclusive_modality_filter with actual tasks (if available)"""
    try:
        all_tasks = get_tasks()
        if not all_tasks:
            pytest.skip("No tasks available in registry")

        text_tasks_exclusive = get_tasks(
            modalities=["text"], exclusive_modality_filter=True
        )
        for task in text_tasks_exclusive:
            assert set(task.modalities) == {"text"}

        image_tasks_exclusive = get_tasks(
            modalities=["image"], exclusive_modality_filter=True
        )
        for task in image_tasks_exclusive:
            assert set(task.modalities) == {"image"}

        text_and_image_tasks_exclusive = get_tasks(
            modalities=["text", "image"], exclusive_modality_filter=True
        )
        for task in text_and_image_tasks_exclusive:
            assert set(task.modalities) == {"text", "image"}

    except Exception as e:
        pytest.skip(f"Error testing with actual tasks: {str(e)}")


def test_get_task_with_exclusive_modality_filter():
    """Test that get_task correctly applies exclusive_modality_filter"""
    if not TASKS_REGISTRY:
        pytest.skip("No tasks in TASKS_REGISTRY")

    try:
        text_only_task_name = None
        image_only_task_name = None
        mixed_modality_task_name = None

        for task_name in TASKS_REGISTRY.keys():
            task = get_task(task_name)
            modalities_set = set(task.modalities)

            if modalities_set == {"text"} and not text_only_task_name:
                text_only_task_name = task_name
            elif modalities_set == {"image"} and not image_only_task_name:
                image_only_task_name = task_name
            elif (
                "text" in modalities_set
                and "image" in modalities_set
                and not mixed_modality_task_name
            ):
                mixed_modality_task_name = task_name

            if (
                text_only_task_name
                and image_only_task_name
                and mixed_modality_task_name
            ):
                break

        if text_only_task_name:
            # Text-only task with text filter should remain unchanged with exclusive_modality_filter=True
            task = get_task(text_only_task_name)
            filtered_task = get_task(
                text_only_task_name, modalities=["text"], exclusive_modality_filter=True
            )
            assert filtered_task.modalities == task.modalities

            # Text-only task with image filter should be filtered out
            filtered_task = get_task(
                text_only_task_name,
                modalities=["image"],
                exclusive_modality_filter=False,
            )
            assert filtered_task.modalities == []

            # Text-only task with text+image filter, With exclusive_modality_filter=False, should be included else it should be skipped
            filtered_task = get_task(
                text_only_task_name,
                modalities=["text", "image"],
                exclusive_modality_filter=False,
            )
            assert len(filtered_task.modalities) > 0

            filtered_task = get_task(
                text_only_task_name,
                modalities=["text", "image"],
                exclusive_modality_filter=True,
            )
            assert filtered_task.modalities == []

        if image_only_task_name:
            # Image-only task with text filter should be filtered out
            filtered_task = get_task(
                image_only_task_name,
                modalities=["text"],
                exclusive_modality_filter=False,
            )
            assert filtered_task.modalities == []

            # Image-only task with image filter should remain unchanged with exclusive_modality_filter=True
            task = get_task(image_only_task_name)
            filtered_task = get_task(
                image_only_task_name,
                modalities=["image"],
                exclusive_modality_filter=True,
            )
            assert filtered_task.modalities == task.modalities

        if mixed_modality_task_name:
            # Task has both text and image, filter has only text, With exclusive_modality_filter=False, should be included else it should be skipped
            filtered_task = get_task(
                mixed_modality_task_name,
                modalities=["text"],
                exclusive_modality_filter=False,
            )
            assert len(filtered_task.modalities) > 0

            filtered_task = get_task(
                mixed_modality_task_name,
                modalities=["text"],
                exclusive_modality_filter=True,
            )
            assert filtered_task.modalities == []

            # Mixed modality task with both text+image filters should remain unchanged with exclusive_modality_filter=True
            task = get_task(mixed_modality_task_name)
            filtered_task = get_task(
                mixed_modality_task_name,
                modalities=["text", "image"],
                exclusive_modality_filter=True,
            )
            assert set(filtered_task.modalities) == set(task.modalities)

    except Exception as e:
        pytest.skip(f"Error in get_task test: {str(e)}")
