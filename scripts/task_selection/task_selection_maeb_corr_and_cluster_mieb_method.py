import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Task Selection via Correlation Threshold

    This notebook performs task selection using pairwise Spearman correlations. Tasks with high average correlations to other tasks are iteratively removed, as they provide redundant information.

    We compare three thresholds (0.7, 0.6, and 0.5) to see how many tasks remain at each level.
    """)
    return


@app.cell
def _():
    from __future__ import annotations

    import mteb

    print(mteb.__version__)
    return (mteb,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Loading in data
    We will start out by loading in the relevant data for the model and tasks of interests.
    """)
    return


@app.cell
def _():
    from pathlib import Path

    results_dir = Path("/Users/isaac/work/maeb-results/results")
    model_names = [
        folder.name.replace("__", "/")
        for folder in results_dir.iterdir()
        if folder.is_dir()
    ]
    len(model_names)
    return (model_names,)


@app.cell
def _(model_names, mteb):
    def get_models():
        models: list[mteb.ModelMeta] = [
            mteb.get_model_meta(name) for name in model_names
        ]

        # get missing revisions - Assuming we are using the latest revision
        for model in models:
            if model.revision is None:
                print(f"Getting revision for {model.name}")
                encoder = model.load_model()
                model.revision = encoder.model_card_data.base_model_revision  # type: ignore

        return models

    models = get_models()

    audio_tasks = mteb.get_tasks(modalities=["audio"])
    len(audio_tasks)
    return audio_tasks, models


@app.cell
def _(audio_tasks):
    # just to see what tasks we are working with
    for _task in audio_tasks:
        print(_task.metadata.name)
        break
    return


@app.cell
def _(audio_tasks, models):
    from mteb.cache import ResultCache

    cache = ResultCache(cache_path="/Users/isaac/work/maeb-results")
    mteb_results = cache.load_results(
        models=models, tasks=audio_tasks, require_model_meta=False
    )
    len(mteb_results.model_results)
    return (mteb_results,)


@app.cell
def _(mteb_results):
    mteb_results.model_results[0]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Creating Results DataFrame
    """)
    return


@app.cell
def _(mteb_results):
    # Set task_name as index, then transpose so rows=models, columns=tasks
    results_df = mteb_results.to_dataframe().set_index("task_name").T
    results_df.head()  # inspect the dataframe
    return (results_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Task Selection

    In this section we will do the task selection using correlation-based filtering.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Manual Curation
    Naturally you can always select your datasets manually and there might be plenty reasons to do so:
    """)
    return


@app.cell
def _():
    # if you wish you can do some manual filtering here, which we will do in this example:
    tasks_to_remove = []
    return (tasks_to_remove,)


@app.cell
def _(audio_tasks, tasks_to_remove):
    # we also want somewhat permissible licenses
    for t in audio_tasks:
        print(t.metadata.name, "-", t.metadata.license)
    # remove tasks with "not specified" licenses
    not_specified_license_tasks = [
        t.metadata.name
        for t in audio_tasks
        if t.metadata.license is None or t.metadata.license.lower() == "not specified"
    ]
    tasks_to_remove_1 = tasks_to_remove + not_specified_license_tasks
    return (tasks_to_remove_1,)


@app.cell
def _(tasks_to_remove_1):
    len(tasks_to_remove_1)
    return


@app.cell
def _(audio_tasks):
    # we also want to removed machine translated datasets
    machine_translated_datasets = [
        t.metadata.name
        for t in audio_tasks
        if t.metadata.sample_creation
        in [
            "machine-translated",
            "machine-translated and verified",
            "machine-translated and localized",
        ]
    ]

    print(machine_translated_datasets)  # there is none
    return


@app.cell
def _(audio_tasks, results_df, tasks_to_remove_1):
    # Filter to tasks not in removal list AND that have results
    tasks_to_select_from = [
        task.metadata.name
        for task in audio_tasks
        if task.metadata.name not in tasks_to_remove_1
        and task.metadata.name in results_df.columns
    ]
    len(tasks_to_select_from)
    return (tasks_to_select_from,)


@app.cell
def _(mteb_results):
    # Build task times per model
    model_task_times = {}
    for _model_result in mteb_results.model_results:
        model_name = _model_result.model_name
        model_task_times[model_name] = {}
        for _task_result in _model_result.task_results:
            if _task_result.evaluation_time:
                model_task_times[model_name][_task_result.task_name] = (
                    _task_result.evaluation_time
                )
    return (model_task_times,)


@app.cell
def _(model_task_times, tasks_to_select_from):
    # Find a CLAP model for reference timing
    clap_models = [m for m in model_task_times.keys() if "clap" in m.lower()]
    if clap_models:
        ref_model = clap_models[0]
        times = model_task_times[ref_model]
        # Filter to tasks we're selecting from
        task_times = [
            (_t, times.get(_t, 0)) for _t in tasks_to_select_from if _t in times
        ]
        top_5 = sorted(task_times, key=lambda x: x[1], reverse=True)[:5]

        print(f"Top 5 longest tasks for {ref_model}:")
        for _task_name, _duration in top_5:
            print(f"  {_task_name}: {_duration / 3600:.2f} hours")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Correlation-Based Task Selection

    We compute pairwise Spearman correlations between tasks (based on model performance), then iteratively remove tasks with high average correlations.
    """)
    return


@app.cell
def _():
    import pandas as pd
    import numpy as np

    return np, pd


@app.cell
def _(mteb, np):
    # tasks which should be kept, e.g. due to them being known high quality datasets, unique tasks, etc.
    tasks_to_keep: list[str] = []

    def compute_correlation_matrix(df, tasks: list[str]):
        """Compute Spearman correlation matrix between tasks."""
        task_df = df[tasks].select_dtypes(include=["number"])
        return task_df.corr(method="spearman")

    def get_avg_correlations(corr_matrix):
        """Get average correlation for each task (excluding self-correlation)."""
        # Set diagonal to NaN to exclude self-correlation
        corr_no_diag = corr_matrix.copy()
        np.fill_diagonal(corr_no_diag.values, np.nan)
        return corr_no_diag.mean(axis=1)

    def get_max_pairwise_correlation(corr_matrix):
        """Find the pair of tasks with highest correlation.

        Returns:
            (task1, task2, correlation) or (None, None, None) if no valid pair
        """
        # Set diagonal to NaN
        corr_no_diag = corr_matrix.copy()
        np.fill_diagonal(corr_no_diag.values, np.nan)

        # Find max correlation
        max_corr = corr_no_diag.max().max()
        if np.isnan(max_corr):
            return None, None, None

        # Find which pair has this correlation
        for _task1 in corr_no_diag.columns:
            for _task2 in corr_no_diag.columns:
                if _task1 != _task2 and corr_no_diag.loc[_task1, _task2] == max_corr:
                    return _task1, _task2, max_corr
        return None, None, None

    def is_candidate_valid_removal(
        current_tasks: list[str], task_to_remove: str
    ) -> bool:
        """Determine if target task should be removed."""
        if task_to_remove in tasks_to_keep:
            return False

        # check if removing task removes a unique task type, category, or domain
        _current_tasks = current_tasks.copy()
        if task_to_remove in _current_tasks:
            _current_tasks.remove(task_to_remove)
        task = mteb.get_task(task_to_remove)
        ctasks = mteb.get_tasks(tasks=_current_tasks)

        # don't remove a unique task type
        task_types = {t.metadata.type for t in ctasks}
        if task.metadata.type not in task_types:
            return False

        # don't remove if it would eliminate a unique category
        if task.metadata.category is not None:
            categories = {
                t.metadata.category for t in ctasks if t.metadata.category is not None
            }
            if task.metadata.category not in categories:
                return False

        # don't remove if it would eliminate a unique domain (domains are multilabel)
        if task.metadata.domains:
            remaining_domains = set()
            for t in ctasks:
                if t.metadata.domains:
                    remaining_domains.update(t.metadata.domains)
            for _domain in task.metadata.domains:
                if _domain not in remaining_domains:
                    return False

        # don't remove if it would eliminate a unique language (languages are multilabel)
        if task.metadata.languages:
            remaining_languages = set()
            for t in ctasks:
                if t.metadata.languages:
                    remaining_languages.update(t.metadata.languages)
            for _lang in task.metadata.languages:
                if _lang not in remaining_languages:
                    return False

        return True

    def iterative_removal_by_correlation(
        df,
        initial_tasks: list[str],
        threshold: float,
    ) -> tuple[list[str], list[str], list[float]]:
        """
        Iteratively remove tasks based on highest pairwise correlation.
        When a pair exceeds threshold, remove the task with higher avg correlation.

        Returns:
            remaining_tasks: Tasks that remain after filtering
            removed_tasks: Tasks removed in order
            max_correlations: Max pairwise correlation at time of removal
        """
        current_tasks = initial_tasks.copy()
        removed_tasks: list[str] = []
        max_correlations: list[float] = []

        while len(current_tasks) > 1:
            corr_matrix = compute_correlation_matrix(df, current_tasks)
            task1, task2, max_corr = get_max_pairwise_correlation(corr_matrix)

            if max_corr is None or max_corr <= threshold:
                break

            # Decide which task to remove: the one with higher avg correlation
            avg_corrs = get_avg_correlations(corr_matrix)

            # Try to remove the one with higher avg correlation first
            candidates = sorted(
                [(task1, avg_corrs[task1]), (task2, avg_corrs[task2])],
                key=lambda x: x[1],
                reverse=True,
            )

            task_removed = False
            for _task_name, _ in candidates:
                if is_candidate_valid_removal(current_tasks, _task_name):
                    current_tasks.remove(_task_name)
                    removed_tasks.append(_task_name)
                    max_correlations.append(max_corr)
                    task_removed = True
                    break

            if not task_removed:
                # Neither task in the pair can be removed
                break

        return current_tasks, removed_tasks, max_correlations

    return compute_correlation_matrix, iterative_removal_by_correlation


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Initial Correlation Matrix

    Let's visualize the correlation matrix before any removal.
    """)
    return


@app.cell
def _(compute_correlation_matrix, results_df, tasks_to_select_from):
    import matplotlib.pyplot as plt
    import seaborn as sns

    initial_corr = compute_correlation_matrix(results_df, tasks_to_select_from)

    fig, ax = plt.subplots(figsize=(45, 40))
    sns.heatmap(initial_corr, annot=False, cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Initial Pairwise Task Correlations (Spearman)", fontsize=18)
    ax.tick_params(axis="both", labelsize=18)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=18)
    plt.tight_layout()
    plt.show()
    return initial_corr, plt


@app.cell
def _(initial_corr):
    # Find tasks with negative correlations to many other tasks
    neg_corr_counts = {}
    for _task in initial_corr.columns:
        # Count how many tasks this task has negative correlation with
        corr_values = initial_corr.loc[_task].drop(_task)  # exclude self
        neg_count = (corr_values < 0).sum()
        if neg_count >= 5:
            avg_neg_corr = corr_values[corr_values < 0].mean()
            neg_corr_counts[_task] = (neg_count, avg_neg_corr)

    # Sort by number of negative correlations (most first)
    sorted_tasks = sorted(neg_corr_counts.items(), key=lambda x: x[1][0], reverse=True)

    print(f"Tasks with 5+ negative correlations: {len(sorted_tasks)}")
    for _task, (_count, _avg) in sorted_tasks:
        print(f"  {_task}: {_count} negative correlations (avg: {_avg:.3f})")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Run Task Selection at Threshold 0.9
    """)
    return


@app.cell
def _(iterative_removal_by_correlation, results_df, tasks_to_select_from):
    remaining_09, removed_09, corrs_09 = iterative_removal_by_correlation(
        results_df, tasks_to_select_from, threshold=0.9
    )
    print(
        f"Threshold 0.9: {len(remaining_09)} tasks remaining, {len(removed_09)} removed"
    )
    return corrs_09, remaining_09, removed_09


@app.cell
def _(removed_09):
    removed_09
    return


@app.cell
def _(remaining_09):
    remaining_09
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Run Task Selection at Threshold 0.8
    """)
    return


@app.cell
def _(iterative_removal_by_correlation, results_df, tasks_to_select_from):
    remaining_08, removed_08, corrs_08 = iterative_removal_by_correlation(
        results_df, tasks_to_select_from, threshold=0.8
    )
    print(
        f"Threshold 0.8: {len(remaining_08)} tasks remaining, {len(removed_08)} removed"
    )
    return corrs_08, remaining_08, removed_08


@app.cell
def _(removed_08):
    removed_08
    return


@app.cell
def _(remaining_08):
    remaining_08
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Run Task Selection at Threshold 0.7
    """)
    return


@app.cell
def _(iterative_removal_by_correlation, results_df, tasks_to_select_from):
    remaining_07, removed_07, corrs_07 = iterative_removal_by_correlation(
        results_df, tasks_to_select_from, threshold=0.7
    )
    print(
        f"Threshold 0.7: {len(remaining_07)} tasks remaining, {len(removed_07)} removed"
    )
    return corrs_07, remaining_07, removed_07


@app.cell
def _(removed_07):
    removed_07
    return


@app.cell
def _(remaining_07):
    remaining_07
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Run Task Selection at Threshold 0.6
    """)
    return


@app.cell
def _(iterative_removal_by_correlation, results_df, tasks_to_select_from):
    remaining_06, removed_06, corrs_06 = iterative_removal_by_correlation(
        results_df, tasks_to_select_from, threshold=0.6
    )
    print(
        f"Threshold 0.6: {len(remaining_06)} tasks remaining, {len(removed_06)} removed"
    )
    return corrs_06, remaining_06, removed_06


@app.cell
def _(removed_06):
    removed_06
    return


@app.cell
def _(remaining_06):
    remaining_06
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Run Task Selection at Threshold 0.5
    """)
    return


@app.cell
def _(iterative_removal_by_correlation, results_df, tasks_to_select_from):
    remaining_05, removed_05, corrs_05 = iterative_removal_by_correlation(
        results_df, tasks_to_select_from, threshold=0.5
    )
    print(
        f"Threshold 0.5: {len(remaining_05)} tasks remaining, {len(removed_05)} removed"
    )
    return corrs_05, remaining_05, removed_05


@app.cell
def _(removed_05):
    removed_05
    return


@app.cell
def _(remaining_05):
    remaining_05
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Comparison of Thresholds
    """)
    return


@app.cell
def _(
    corrs_05,
    corrs_06,
    corrs_07,
    corrs_08,
    corrs_09,
    plt,
    removed_05,
    removed_06,
    removed_07,
    removed_08,
    removed_09,
):
    fig2, axes = plt.subplots(1, 5, figsize=(30, 5))

    thresholds = [0.9, 0.8, 0.7, 0.6, 0.5]
    removed_lists = [removed_09, removed_08, removed_07, removed_06, removed_05]
    corrs_lists = [corrs_09, corrs_08, corrs_07, corrs_06, corrs_05]

    for _idx, (_thresh, _removed, _corrs) in enumerate(
        zip(thresholds, removed_lists, corrs_lists)
    ):
        if _removed:
            axes[_idx].plot(range(len(_corrs)), _corrs, marker="o")
            axes[_idx].axhline(y=_thresh, color="r", linestyle="--", label="Threshold")
            axes[_idx].set_xlabel("Removal step")
            axes[_idx].set_ylabel("Max pairwise correlation at removal")
            axes[_idx].set_title(f"Threshold {_thresh}: {len(_removed)} tasks removed")
            axes[_idx].set_xticks(range(len(_removed)))
            axes[_idx].set_xticklabels(_removed, rotation=90)
            axes[_idx].legend()
        else:
            axes[_idx].text(0.5, 0.5, "No tasks removed", ha="center", va="center")
            axes[_idx].set_title(f"Threshold {_thresh}")

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(remaining_05, remaining_06, remaining_07, remaining_08, remaining_09):
    # Tasks in all thresholds (most restrictive)
    in_all = (
        set(remaining_09)
        & set(remaining_08)
        & set(remaining_07)
        & set(remaining_06)
        & set(remaining_05)
    )

    print(f"Tasks in all five thresholds: {len(in_all)}")
    print(f"Remaining at 0.9: {len(remaining_09)}")
    print(f"Remaining at 0.8: {len(remaining_08)}")
    print(f"Remaining at 0.7: {len(remaining_07)}")
    print(f"Remaining at 0.6: {len(remaining_06)}")
    print(f"Remaining at 0.5: {len(remaining_05)}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Collection Comparison

    Compare the filtered collections against the full collection in terms of:
    1. Correlation of average model performance
    2. Total evaluation runtime
    """)
    return


@app.cell
def _(
    remaining_05,
    remaining_06,
    remaining_07,
    remaining_08,
    remaining_09,
    results_df,
    tasks_to_select_from,
):
    # Compute average performance per model for each collection
    full_avg = results_df[tasks_to_select_from].mean(axis=1)
    avg_09 = results_df[remaining_09].mean(axis=1)
    avg_08 = results_df[remaining_08].mean(axis=1)
    avg_07 = results_df[remaining_07].mean(axis=1)
    avg_06 = results_df[remaining_06].mean(axis=1)
    avg_05 = results_df[remaining_05].mean(axis=1)
    return avg_05, avg_06, avg_07, avg_08, avg_09, full_avg


@app.cell
def _(avg_05, avg_06, avg_07, avg_08, avg_09, full_avg):
    from scipy.stats import spearmanr, pearsonr

    # Full vs 0.9 threshold
    spearman_09 = spearmanr(full_avg, avg_09)[0]
    pearson_09 = pearsonr(full_avg, avg_09)[0]

    # Full vs 0.8 threshold
    spearman_08 = spearmanr(full_avg, avg_08)[0]
    pearson_08 = pearsonr(full_avg, avg_08)[0]

    # Full vs 0.7 threshold
    spearman_07 = spearmanr(full_avg, avg_07)[0]
    pearson_07 = pearsonr(full_avg, avg_07)[0]

    # Full vs 0.6 threshold
    spearman_06 = spearmanr(full_avg, avg_06)[0]
    pearson_06 = pearsonr(full_avg, avg_06)[0]

    # Full vs 0.5 threshold
    spearman_05 = spearmanr(full_avg, avg_05)[0]
    pearson_05 = pearsonr(full_avg, avg_05)[0]

    print("Correlation of average model performance (filtered vs full):")
    print(f"  Threshold 0.9:  Spearman={spearman_09:.4f}, Pearson={pearson_09:.4f}")
    print(f"  Threshold 0.8:  Spearman={spearman_08:.4f}, Pearson={pearson_08:.4f}")
    print(f"  Threshold 0.7:  Spearman={spearman_07:.4f}, Pearson={pearson_07:.4f}")
    print(f"  Threshold 0.6:  Spearman={spearman_06:.4f}, Pearson={pearson_06:.4f}")
    print(f"  Threshold 0.5:  Spearman={spearman_05:.4f}, Pearson={pearson_05:.4f}")
    return (
        pearson_05,
        pearson_06,
        pearson_07,
        pearson_08,
        pearson_09,
        spearman_05,
        spearman_06,
        spearman_07,
        spearman_08,
        spearman_09,
    )


@app.cell
def _(model_task_times, mteb, tasks_to_select_from):
    # Filter to models with valid eval times for ALL tasks in the collection
    tasks_set = set(tasks_to_select_from)
    valid_models = [
        _model_name
        for _model_name, times in model_task_times.items()
        if tasks_set.issubset(set(times.keys()))
    ]
    print(
        f"Models with complete eval times for all {len(tasks_to_select_from)} tasks: {len(valid_models)}, {valid_models}"
    )

    # Find largest and smallest models by memory usage (MB) among valid models
    model_memory = {}
    for _model_name in valid_models:
        _meta = mteb.get_model_meta(_model_name)
        if _meta.memory_usage_mb:
            model_memory[_model_name] = _meta.memory_usage_mb

    largest_model = max(model_memory, key=model_memory.get)
    smallest_model = min(model_memory, key=model_memory.get)

    print(
        f"Largest model (by memory):  {largest_model} ({model_memory[largest_model]:.1f} MB)"
    )
    print(
        f"Smallest model (by memory): {smallest_model} ({model_memory[smallest_model]:.1f} MB)"
    )
    return largest_model, smallest_model


@app.cell
def _(
    largest_model,
    model_task_times,
    remaining_05,
    remaining_06,
    remaining_07,
    remaining_08,
    remaining_09,
    smallest_model,
    tasks_to_select_from,
):
    # Compute runtime for each collection using largest and smallest models
    def compute_runtime_hours(model_name, task_list):
        times = model_task_times.get(model_name, {})
        return sum(times.get(t, 0) for t in task_list) / 3600

    # Largest model runtimes
    runtime_large_full = compute_runtime_hours(largest_model, tasks_to_select_from)
    runtime_large_09 = compute_runtime_hours(largest_model, remaining_09)
    runtime_large_08 = compute_runtime_hours(largest_model, remaining_08)
    runtime_large_07 = compute_runtime_hours(largest_model, remaining_07)
    runtime_large_06 = compute_runtime_hours(largest_model, remaining_06)
    runtime_large_05 = compute_runtime_hours(largest_model, remaining_05)

    # Smallest model runtimes
    runtime_small_full = compute_runtime_hours(smallest_model, tasks_to_select_from)
    runtime_small_09 = compute_runtime_hours(smallest_model, remaining_09)
    runtime_small_08 = compute_runtime_hours(smallest_model, remaining_08)
    runtime_small_07 = compute_runtime_hours(smallest_model, remaining_07)
    runtime_small_06 = compute_runtime_hours(smallest_model, remaining_06)
    runtime_small_05 = compute_runtime_hours(smallest_model, remaining_05)

    print(f"Runtime for LARGEST model ({largest_model}):")
    print(f"  Full:           {runtime_large_full:.3f} hours")
    print(f"  Threshold 0.9:  {runtime_large_09:.3f} hours")
    print(f"  Threshold 0.8:  {runtime_large_08:.3f} hours")
    print(f"  Threshold 0.7:  {runtime_large_07:.3f} hours")
    print(f"  Threshold 0.6:  {runtime_large_06:.3f} hours")
    print(f"  Threshold 0.5:  {runtime_large_05:.3f} hours")
    print()
    print(f"Runtime for SMALLEST model ({smallest_model}):")
    print(f"  Full:           {runtime_small_full:.3f} hours")
    print(f"  Threshold 0.9:  {runtime_small_09:.3f} hours")
    print(f"  Threshold 0.8:  {runtime_small_08:.3f} hours")
    print(f"  Threshold 0.7:  {runtime_small_07:.3f} hours")
    print(f"  Threshold 0.6:  {runtime_small_06:.3f} hours")
    print(f"  Threshold 0.5:  {runtime_small_05:.3f} hours")
    return (
        runtime_large_05,
        runtime_large_06,
        runtime_large_07,
        runtime_large_08,
        runtime_large_09,
        runtime_large_full,
        runtime_small_05,
        runtime_small_06,
        runtime_small_07,
        runtime_small_08,
        runtime_small_09,
        runtime_small_full,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary
    """)
    return


@app.cell
def _(
    largest_model,
    pd,
    pearson_05,
    pearson_06,
    pearson_07,
    pearson_08,
    pearson_09,
    remaining_05,
    remaining_06,
    remaining_07,
    remaining_08,
    remaining_09,
    runtime_large_05,
    runtime_large_06,
    runtime_large_07,
    runtime_large_08,
    runtime_large_09,
    runtime_large_full,
    runtime_small_05,
    runtime_small_06,
    runtime_small_07,
    runtime_small_08,
    runtime_small_09,
    runtime_small_full,
    smallest_model,
    spearman_05,
    spearman_06,
    spearman_07,
    spearman_08,
    spearman_09,
    tasks_to_select_from,
):
    # Enhanced summary table
    comparison_data = {
        "Collection": [
            "Full",
            "Threshold 0.9",
            "Threshold 0.8",
            "Threshold 0.7",
            "Threshold 0.6",
            "Threshold 0.5",
        ],
        "Tasks": [
            len(tasks_to_select_from),
            len(remaining_09),
            len(remaining_08),
            len(remaining_07),
            len(remaining_06),
            len(remaining_05),
        ],
        "Spearman (vs Full)": [
            1.0,
            spearman_09,
            spearman_08,
            spearman_07,
            spearman_06,
            spearman_05,
        ],
        "Pearson (vs Full)": [
            1.0,
            pearson_09,
            pearson_08,
            pearson_07,
            pearson_06,
            pearson_05,
        ],
        "Runtime-Large (h)": [
            f"{runtime_large_full:.3f}",
            f"{runtime_large_09:.3f}",
            f"{runtime_large_08:.3f}",
            f"{runtime_large_07:.3f}",
            f"{runtime_large_06:.3f}",
            f"{runtime_large_05:.3f}",
        ],
        "Runtime-Small (h)": [
            f"{runtime_small_full:.3f}",
            f"{runtime_small_09:.3f}",
            f"{runtime_small_08:.3f}",
            f"{runtime_small_07:.3f}",
            f"{runtime_small_06:.3f}",
            f"{runtime_small_05:.3f}",
        ],
    }
    comparison_df = pd.DataFrame(comparison_data)
    print(f"Large model: {largest_model}")
    print(f"Small model: {smallest_model}")
    comparison_df
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Task Clustering via UMAP + HDBSCAN

    We cluster tasks using their correlation vectors (each task's row in the correlation matrix).
    UMAP reduces dimensionality, then HDBSCAN identifies clusters.

    We expect the outlier cluster (-1 label) to span all task categories, serving as a foundation for balanced selection.
    """)
    return


@app.cell
def _():
    import umap
    import hdbscan

    return hdbscan, umap


@app.cell
def _(hdbscan, initial_corr, np, umap):
    # Use correlation vectors as features (each task's correlations with all other tasks)
    correlation_vectors = initial_corr.values

    # Handle NaN values by filling with 0 (no correlation)
    correlation_vectors_clean = np.nan_to_num(correlation_vectors, nan=0.0)

    # UMAP dimensionality reduction
    umap_reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric="euclidean",
        random_state=42,
    )
    umap_embedding = umap_reducer.fit_transform(correlation_vectors_clean)

    # HDBSCAN clustering
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=3, min_samples=2, cluster_selection_epsilon=0.0
    )
    cluster_labels = clusterer.fit_predict(umap_embedding)

    print(
        f"Number of clusters (excluding outliers): {len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)}"
    )
    print(f"Number of outliers (label -1): {sum(cluster_labels == -1)}")
    return cluster_labels, umap_embedding


@app.cell
def _(cluster_labels, initial_corr, np, plt, umap_embedding):
    # Create a DataFrame with task names and their cluster assignments
    task_cluster_map = dict(zip(initial_corr.columns, cluster_labels))

    # Visualize the UMAP embedding with cluster colors
    fig_umap, ax_umap = plt.subplots(figsize=(12, 10))

    unique_labels = sorted(set(cluster_labels))
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

    for _idx, _label in enumerate(unique_labels):
        mask = cluster_labels == _label
        if _label == -1:
            ax_umap.scatter(
                umap_embedding[mask, 0],
                umap_embedding[mask, 1],
                c="gray",
                marker="x",
                s=100,
                label=f"Outliers ({sum(mask)})",
                alpha=0.7,
            )
        else:
            ax_umap.scatter(
                umap_embedding[mask, 0],
                umap_embedding[mask, 1],
                c=[colors[_idx]],
                marker="o",
                s=100,
                label=f"Cluster {_label} ({sum(mask)})",
                alpha=0.7,
            )

    ax_umap.set_xlabel("UMAP 1", fontsize=12)
    ax_umap.set_ylabel("UMAP 2", fontsize=12)
    ax_umap.set_title("Task Clustering via UMAP + HDBSCAN", fontsize=14)
    ax_umap.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(cluster_labels, initial_corr, mteb, pd):
    # Analyze task types in each cluster
    tasks_list = list(initial_corr.columns)
    cluster_analysis = []

    for _task_name, _label in zip(tasks_list, cluster_labels):
        _task = mteb.get_task(_task_name)
        cluster_analysis.append(
            {"task": _task_name, "cluster": _label, "task_type": _task.metadata.type}
        )

    cluster_df = pd.DataFrame(cluster_analysis)
    cluster_df
    return (cluster_df,)


@app.cell
def _(cluster_df):
    # Show task type distribution per cluster
    cluster_type_counts = (
        cluster_df.groupby(["cluster", "task_type"]).size().unstack(fill_value=0)
    )
    print("Task type distribution per cluster:")
    cluster_type_counts
    return


@app.cell
def _(cluster_df):
    # Focus on outlier cluster (-1)
    outlier_tasks = cluster_df[cluster_df["cluster"] == -1]
    print(f"Outlier cluster contains {len(outlier_tasks)} tasks spanning these types:")
    print(outlier_tasks["task_type"].value_counts())
    print("\nOutlier tasks:")
    outlier_tasks
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
