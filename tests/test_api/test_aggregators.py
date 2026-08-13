from mteb.api.aggregators import (
    _recompute_lenient_custom_groups,
    _recompute_lenient_means,
)


def test_recompute_lenient_custom_groups_averages_present_tasks_only():
    """Mirrors _recompute_lenient_means: only tasks the model actually has a
    score for (post language-filter) contribute to each group's mean."""
    scores_by_task = {"t1": 0.2, "t2": 0.6, "t3": 0.8}
    custom_group_task_to_label = {
        "DimA": {"t1": "G1", "t2": "G1", "t3": "Other"},
        "DimB": {"t2": "G2", "t3": "Other"},
    }

    out = _recompute_lenient_custom_groups(scores_by_task, custom_group_task_to_label)

    assert out == {
        "DimA": {"G1": (0.2 + 0.6) / 2, "Other": 0.8},
        "DimB": {"G2": 0.6, "Other": 0.8},
    }


def test_recompute_lenient_custom_groups_ignores_tasks_outside_the_visible_set():
    """A task missing from scores_by_task (filtered out / not run) is simply
    absent from its group's bucket, not treated as a zero."""
    scores_by_task = {"t1": 1.0}
    custom_group_task_to_label = {"DimA": {"t1": "G1", "t2": "G1"}}

    out = _recompute_lenient_custom_groups(scores_by_task, custom_group_task_to_label)

    assert out == {"DimA": {"G1": 1.0}}


def test_recompute_lenient_custom_groups_empty_mapping_is_a_no_op():
    """Benchmarks with no CustomGrouping declared pass an empty mapping —
    the function must return {} rather than raise."""
    assert _recompute_lenient_custom_groups({"t1": 0.5}, {}) == {}


def test_recompute_lenient_custom_groups_matches_recompute_lenient_means_semantics():
    """Same 'average only what's present' policy as the task-type recompute,
    just keyed by CustomGrouping label instead of task type."""
    scores_by_task = {"t1": 0.4, "t2": 0.9}
    task_to_type = {"t1": "Retrieval", "t2": "Retrieval"}
    custom_group_task_to_label = {"DimA": {"t1": "G1", "t2": "G1"}}

    _, mean_task, _ = _recompute_lenient_means(scores_by_task, task_to_type)
    out = _recompute_lenient_custom_groups(scores_by_task, custom_group_task_to_label)

    assert out["DimA"]["G1"] == mean_task
