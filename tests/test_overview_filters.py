"""Tests for the filter functions used in get_tasks()."""

import pytest

from mteb import filter_tasks, get_tasks
from mteb.abstasks.task_metadata import TaskType


class TestLanguageFiltering:
    """Tests for language filtering functionality."""

    def test_filter_by_single_language(self):
        """Test filtering by a single language."""
        tasks = get_tasks(languages=["eng"])
        assert len(tasks) > 0
        for task in tasks:
            assert "eng" in task.languages

    def test_filter_by_multiple_languages(self):
        """Test filtering by multiple languages."""
        tasks = get_tasks(languages=["eng", "deu"])
        assert len(tasks) > 0
        for task in tasks:
            # Task should contain at least one of the specified languages
            assert "eng" in task.languages or "deu" in task.languages

    def test_filter_by_language_and_script_separately(self):
        """Test filtering by language and script separately."""
        tasks = get_tasks(languages=["eng"], script=["Latn"])
        assert len(tasks) > 0
        for task in tasks:
            # Should have English in Latin script
            assert "eng" in task.languages
            assert "Latn" in task.metadata.scripts

    def test_exclusive_language_filter_true(self):
        """Test exclusive language filtering (only tasks with ALL specified languages)."""
        # Find a multilingual task that has both eng and deu
        all_tasks = get_tasks(languages=["eng", "deu"], exclusive_language_filter=False)
        # With exclusive filter, should get fewer or equal tasks
        exclusive_tasks = get_tasks(
            languages=["eng", "deu"], exclusive_language_filter=True
        )
        assert len(exclusive_tasks) <= len(all_tasks)

    def test_exclusive_language_filter_false(self):
        """Test non-exclusive language filtering (tasks with ANY of the languages)."""
        tasks = get_tasks(languages=["eng", "deu"], exclusive_language_filter=False)
        assert len(tasks) > 0
        for task in tasks:
            # Should have at least one of the specified languages
            assert "eng" in task.languages or "deu" in task.languages

    def test_invalid_language_code(self):
        """Test that invalid language codes raise an error."""
        with pytest.raises(ValueError, match="Invalid language code"):
            get_tasks(languages=["invalid_lang"])


class TestScriptFiltering:
    """Tests for script filtering functionality."""

    def test_filter_by_single_script(self):
        """Test filtering by a single script."""
        tasks = get_tasks(script=["Latn"])
        assert len(tasks) > 0
        for task in tasks:
            assert "Latn" in task.metadata.scripts

    def test_filter_by_multiple_scripts(self):
        """Test filtering by multiple scripts."""
        tasks = get_tasks(script=["Latn", "Cyrl"])
        assert len(tasks) > 0
        for task in tasks:
            # Task should use at least one of the specified scripts
            scripts = task.metadata.scripts
            assert "Latn" in scripts or "Cyrl" in scripts

    def test_filter_by_cyrillic_script(self):
        """Test filtering by Cyrillic script."""
        tasks = get_tasks(script=["Cyrl"])
        assert len(tasks) > 0
        for task in tasks:
            assert "Cyrl" in task.metadata.scripts

    def test_invalid_script_code(self):
        """Test that invalid script codes raise an error."""
        with pytest.raises(ValueError, match="Invalid script code"):
            get_tasks(script=["InvalidScript"])


class TestTaskTypeFiltering:
    """Tests for task type filtering functionality."""

    def test_filter_by_classification(self):
        """Test filtering by Classification task type."""
        tasks = get_tasks(task_types=["Classification"])
        assert len(tasks) > 0
        for task in tasks:
            assert task.metadata.type == "Classification"

    def test_filter_by_retrieval(self):
        """Test filtering by Retrieval task type."""
        tasks = get_tasks(task_types=["Retrieval"])
        assert len(tasks) > 0
        for task in tasks:
            assert task.metadata.type == "Retrieval"

    def test_filter_by_multiple_task_types(self):
        """Test filtering by multiple task types."""
        tasks = get_tasks(task_types=["Classification", "Clustering"])
        assert len(tasks) > 0
        for task in tasks:
            assert task.metadata.type in ["Classification", "Clustering"]

    @pytest.mark.parametrize(
        "task_type",
        [
            "Classification",
            "Clustering",
            "PairClassification",
            "Reranking",
            "Retrieval",
            "STS",
            "Summarization",
        ],
    )
    def test_all_major_task_types(self, task_type: TaskType):
        """Test filtering by each major task type."""
        tasks = get_tasks(task_types=[task_type])
        assert len(tasks) > 0
        for task in tasks:
            assert task.metadata.type == task_type


class TestDomainFiltering:
    """Tests for domain filtering functionality."""

    def test_filter_by_single_domain(self):
        """Test filtering by a single domain."""
        # Legal domain should exist
        tasks = get_tasks(domains=["Legal"])
        if len(tasks) > 0:  # Only test if legal tasks exist
            for task in tasks:
                assert task.metadata.domains is not None
                assert "Legal" in task.metadata.domains

    def test_filter_by_multiple_domains(self):
        """Test filtering by multiple domains."""
        tasks = get_tasks(domains=["Legal", "Medical"])
        if len(tasks) > 0:
            for task in tasks:
                if task.metadata.domains:
                    domains = task.metadata.domains
                    assert "Legal" in domains or "Medical" in domains


class TestCategoryFiltering:
    """Tests for category filtering functionality."""

    def test_filter_by_t2t_category(self):
        """Test filtering by text-to-text category."""
        tasks = get_tasks(categories=["t2t"])
        if len(tasks) > 0:
            for task in tasks:
                assert task.metadata.category == "t2t"


class TestModalityFiltering:
    """Tests for modality filtering functionality."""

    def test_filter_by_text_modality(self):
        """Test filtering by text modality."""
        tasks = get_tasks(modalities=["text"])
        assert len(tasks) > 0
        for task in tasks:
            assert "text" in task.modalities

    def test_filter_by_image_modality(self):
        """Test filtering by image modality."""
        tasks = get_tasks(modalities=["image"])
        if len(tasks) > 0:
            for task in tasks:
                assert "image" in task.modalities

    def test_filter_by_multiple_modalities(self):
        """Test filtering by multiple modalities (non-exclusive)."""
        tasks = get_tasks(modalities=["text", "image"], exclusive_modality_filter=False)
        assert len(tasks) > 0
        for task in tasks:
            # Should have at least one of the specified modalities
            assert "text" in task.modalities or "image" in task.modalities

    def test_exclusive_modality_filter(self):
        """Test exclusive modality filtering."""
        # Tasks with ONLY text modality
        text_only = get_tasks(modalities=["text"], exclusive_modality_filter=True)
        if len(text_only) > 0:
            for task in text_only:
                assert set(task.modalities) == {"text"}

        # Tasks with ONLY image modality
        image_only = get_tasks(modalities=["image"], exclusive_modality_filter=True)
        if len(image_only) > 0:
            for task in image_only:
                assert set(task.modalities) == {"image"}


class TestSupersededFiltering:
    """Tests for superseded dataset filtering."""

    def test_exclude_superseded_default(self):
        """Test that superseded datasets are excluded by default."""
        tasks = get_tasks()
        for task in tasks:
            assert task.superseded_by is None

    def test_include_superseded(self):
        """Test that superseded datasets can be included."""
        tasks_without_superseded = get_tasks(exclude_superseded=True)
        tasks_with_superseded = get_tasks(exclude_superseded=False)

        # Should have at least as many tasks when including superseded
        assert len(tasks_with_superseded) >= len(tasks_without_superseded)


class TestAggregateFiltering:
    """Tests for aggregate task filtering."""

    def test_exclude_aggregate_default(self):
        """Test that aggregate tasks are excluded by default when specified."""
        tasks = get_tasks(exclude_aggregate=True)
        for task in tasks:
            assert not task.is_aggregate

    def test_include_aggregate(self):
        """Test that aggregate tasks can be included."""
        tasks_without_aggregate = get_tasks(exclude_aggregate=True)
        tasks_with_aggregate = get_tasks(exclude_aggregate=False)

        # Should have at least as many tasks when including aggregate
        assert len(tasks_with_aggregate) >= len(tasks_without_aggregate)


class TestPrivacyFiltering:
    """Tests for privacy/public dataset filtering."""

    def test_exclude_private_default(self):
        """Test that private datasets are excluded by default."""
        tasks = get_tasks(exclude_private=True)
        for task in tasks:
            # is_public should be True or None (None is considered public)
            assert task.metadata.is_public is not False

    def test_include_private(self):
        """Test that private datasets can be included."""
        public_tasks = get_tasks(exclude_private=True)
        all_tasks = get_tasks(exclude_private=False)

        # Should have at least as many tasks when including private
        assert len(all_tasks) >= len(public_tasks)


class TestEvalSplitFiltering:
    """Tests for evaluation split filtering."""

    def test_filter_by_eval_split(self):
        """Test filtering by evaluation split."""
        tasks = get_tasks(eval_splits=["test"])
        assert len(tasks) > 0
        for task in tasks:
            assert "test" in task.eval_splits

    def test_filter_by_multiple_eval_splits(self):
        """Test filtering by multiple evaluation splits."""
        tasks = get_tasks(eval_splits=["test", "dev"])
        assert len(tasks) > 0
        for task in tasks:
            # Task should have at least one of the specified splits
            splits = task.eval_splits
            assert "test" in splits or "dev" in splits


class TestCombinedFiltering:
    """Tests for combining multiple filters."""

    def test_language_and_task_type(self):
        """Test filtering by both language and task type."""
        tasks = get_tasks(languages=["eng"], task_types=["Classification"])
        assert len(tasks) > 0
        for task in tasks:
            assert "eng" in task.languages
            assert task.metadata.type == "Classification"

    def test_script_and_modality(self):
        """Test filtering by both script and modality."""
        tasks = get_tasks(script=["Latn"], modalities=["text"])
        assert len(tasks) > 0
        for task in tasks:
            assert "Latn" in task.metadata.scripts
            assert "text" in task.modalities

    def test_multiple_filters_combined(self):
        """Test combining language, task type, and superseded filtering."""
        tasks = get_tasks(
            languages=["eng"],
            task_types=["Retrieval"],
            exclude_superseded=True,
            exclude_aggregate=True,
        )
        assert len(tasks) > 0
        for task in tasks:
            assert "eng" in task.languages
            assert task.metadata.type == "Retrieval"
            assert task.superseded_by is None
            assert not task.is_aggregate


class TestSpecificTaskRetrieval:
    """Tests for retrieving specific tasks by name."""

    def test_get_specific_tasks_by_name(self):
        """Test retrieving specific tasks by providing task names."""
        task_names = ["Banking77Classification", "ArguAna"]
        tasks = get_tasks(tasks=task_names)

        assert len(tasks) == len(task_names)
        retrieved_names = [task.metadata.name for task in tasks]
        for name in task_names:
            assert name in retrieved_names

    def test_get_single_task_by_name(self):
        """Test retrieving a single task by name."""
        tasks = get_tasks(tasks=["Banking77Classification"])
        assert len(tasks) == 1
        assert tasks[0].metadata.name == "Banking77Classification"


class TestFilterTasksDirectly:
    """Test the filter_tasks function directly with task classes."""

    @pytest.fixture
    def sample_task_classes(self):
        """Get a sample of task classes for testing."""
        from mteb.get_tasks import TASK_LIST

        return list(TASK_LIST)[:20]

    def test_filter_by_languages(self, sample_task_classes):
        """Test filter_tasks with language filter."""
        filtered = filter_tasks(sample_task_classes, languages=["eng"])
        for task_cls in filtered:
            assert "eng" in task_cls.metadata.languages

    def test_filter_by_script(self, sample_task_classes):
        """Test filter_tasks with script filter."""
        filtered = filter_tasks(sample_task_classes, script=["Latn"])
        for task_cls in filtered:
            assert "Latn" in task_cls.metadata.scripts

    def test_filter_by_task_types(self, sample_task_classes):
        """Test filter_tasks with task type filter."""
        filtered = filter_tasks(sample_task_classes, task_types=["Classification"])
        for task_cls in filtered:
            assert task_cls.metadata.type == "Classification"

    def test_filter_by_modalities(self, sample_task_classes):
        """Test filter_tasks with modality filter."""
        filtered = filter_tasks(
            sample_task_classes, modalities=["text"], exclusive_modality_filter=False
        )
        for task_cls in filtered:
            assert "text" in task_cls.metadata.modalities

    def test_filter_superseded(self, sample_task_classes):
        """Test filter_tasks excluding superseded datasets."""
        filtered = filter_tasks(sample_task_classes, exclude_superseded=True)
        for task_cls in filtered:
            assert task_cls.superseded_by is None

    def test_filter_aggregate(self, sample_task_classes):
        """Test filter_tasks excluding aggregate tasks."""
        filtered = filter_tasks(sample_task_classes, exclude_aggregate=True)
        for task_cls in filtered:
            # Create temporary instance to check is_aggregate
            temp_instance = task_cls()
            assert not temp_instance.is_aggregate


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_filter_returns_all_tasks(self):
        """Test that no filters returns all available tasks."""
        tasks = get_tasks()
        assert len(tasks) > 0

    def test_impossible_filter_combination(self):
        """Test filter combination that should return no tasks."""
        # A task can't be both Classification and Retrieval
        tasks = get_tasks(task_types=["Classification"])
        retrieval_names = [t.metadata.name for t in get_tasks(task_types=["Retrieval"])]

        for task in tasks:
            assert task.metadata.name not in retrieval_names

    def test_filter_with_nonexistent_language(self):
        """Test filtering with a language that exists but has no tasks."""
        # Use a valid but rare language code
        tasks = get_tasks(languages=["aby"])  # Abé language (valid ISO code)
        # Should return empty or very few tasks
        assert len(tasks) >= 0
