import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell
def _():
    from __future__ import annotations

    import json
    from pathlib import Path

    import pandas as pd

    from mteb.benchmarks import get_benchmark

    return Path, get_benchmark, json, pd


@app.cell
def _(Path):
    RESULTS_DIR = Path("/Users/isaac/work/maeb-results/results")
    BENCHMARK_NAMES = [
        "MAEB(audio-only)",
        "MAEB",
        "MAEB(extended)",
    ]
    return BENCHMARK_NAMES, RESULTS_DIR


@app.cell
def _(BENCHMARK_NAMES, get_benchmark):
    benchmarks = {name: get_benchmark(name) for name in BENCHMARK_NAMES}

    benchmark_tasks = {
        name: [task.metadata.name for task in bench.tasks]
        for name, bench in benchmarks.items()
    }

    for name, task_list in benchmark_tasks.items():
        print(f"{name}: {len(task_list)} tasks")
    return (benchmark_tasks,)


@app.cell
def _(Path, json):
    def load_model_meta(revision_folder: Path) -> dict | None:
        """Load model_meta.json from a revision folder."""
        meta_path = revision_folder / "model_meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                return json.load(f)
        return None

    def load_eval_times(
        revision_folder: Path, task_names: list[str]
    ) -> dict[str, float]:
        """Load evaluation_time from task JSON files."""
        eval_times = {}
        for task_name in task_names:
            task_file = revision_folder / f"{task_name}.json"
            if task_file.exists():
                try:
                    with open(task_file) as f:
                        data = json.load(f)
                    if (
                        "evaluation_time" in data
                        and data["evaluation_time"] is not None
                    ):
                        eval_times[task_name] = data["evaluation_time"]
                except (json.JSONDecodeError, KeyError):
                    pass
        return eval_times

    return load_eval_times, load_model_meta


@app.cell
def _(RESULTS_DIR, benchmark_tasks, load_eval_times, load_model_meta):
    # Collect data for all models across all benchmarks
    model_data = []

    for model_folder in RESULTS_DIR.iterdir():
        if not model_folder.is_dir():
            continue

        model_name = model_folder.name.replace("__", "/")

        for revision_folder in model_folder.iterdir():
            if not revision_folder.is_dir():
                continue

            revision = revision_folder.name
            meta = load_model_meta(revision_folder)

            if meta is None:
                continue

            memory_mb = meta.get("memory_usage_mb")
            n_params = meta.get("n_parameters")

            # Skip models without size information
            if memory_mb is None and n_params is None:
                continue

            record = {
                "model_name": model_name,
                "revision": revision,
                "memory_usage_mb": memory_mb,
                "n_parameters": n_params,
            }

            # Load eval times for each benchmark
            for bench_name, bench_tasks in benchmark_tasks.items():
                eval_times = load_eval_times(revision_folder, bench_tasks)
                total_time = sum(eval_times.values()) if eval_times else None
                task_count = len(eval_times)
                record[f"{bench_name}_eval_time"] = total_time
                record[f"{bench_name}_task_count"] = task_count

            model_data.append(record)

    print(f"Loaded data for {len(model_data)} model revisions")
    return (model_data,)


@app.cell
def _(model_data, pd):
    # Create DataFrame from collected data
    df_all_models = pd.DataFrame(model_data)

    # Use memory_usage_mb if available, otherwise estimate from n_parameters (4 bytes per param)
    df_all_models["size_mb"] = df_all_models["memory_usage_mb"].fillna(
        df_all_models["n_parameters"] * 4 / 1e6
    )

    # Filter to only models that have size information
    df_with_size = df_all_models[df_all_models["size_mb"].notna()].copy()

    print(f"Models with size info: {len(df_with_size)}")
    df_with_size.head()
    return (df_with_size,)


@app.cell
def _(BENCHMARK_NAMES, df_with_size):
    # For each benchmark, find top 5 largest and top 5 smallest models
    top_models_per_benchmark = {}

    for _bn in BENCHMARK_NAMES:
        _tcol = f"{_bn}_eval_time"
        _ccol = f"{_bn}_task_count"

        # Filter to models that have results for this benchmark
        _df_bench = df_with_size[df_with_size[_tcol].notna()].copy()

        if len(_df_bench) == 0:
            print(f"No models with results for {_bn}")
            continue

        # Sort by size
        _df_sorted = _df_bench.sort_values("size_mb", ascending=False)

        # Get unique models by name (take first revision for each)
        _df_unique = _df_sorted.drop_duplicates(subset=["model_name"], keep="first")

        _top_5_largest = set(_df_unique.head(5)["model_name"].tolist())
        _top_5_smallest = set(_df_unique.tail(5)["model_name"].tolist())

        top_models_per_benchmark[_bn] = {
            "largest": _top_5_largest,
            "smallest": _top_5_smallest,
        }

        print(f"\n{_bn}:")
        print(f"  Top 5 largest: {_top_5_largest}")
        print(f"  Top 5 smallest: {_top_5_smallest}")
    return (top_models_per_benchmark,)


@app.cell
def _(top_models_per_benchmark):
    # Find common models between MAEB_AUDIO and MAEB
    _audio_benchmarks = ["MAEB(audio-only)", "MAEB"]
    _extended_benchmark = "MAEB(extended)"

    # Get sets for audio benchmarks
    _largest_sets = [
        top_models_per_benchmark[b]["largest"]
        for b in _audio_benchmarks
        if b in top_models_per_benchmark
    ]
    _smallest_sets = [
        top_models_per_benchmark[b]["smallest"]
        for b in _audio_benchmarks
        if b in top_models_per_benchmark
    ]

    # Find common models between MAEB(audio-only) and MAEB
    common_audio_largest = (
        set.intersection(*_largest_sets) if len(_largest_sets) == 2 else set()
    )
    common_audio_smallest = (
        set.intersection(*_smallest_sets) if len(_smallest_sets) == 2 else set()
    )

    print(f"Common large models (MAEB audio-only & MAEB): {common_audio_largest}")
    print(f"Common small models (MAEB audio-only & MAEB): {common_audio_smallest}")

    # Select one from each for audio benchmarks
    selected_audio_large = (
        list(common_audio_largest)[0] if common_audio_largest else None
    )
    selected_audio_small = (
        list(common_audio_smallest)[0] if common_audio_smallest else None
    )

    print(f"\nSelected audio large model: {selected_audio_large}")
    print(f"Selected audio small model: {selected_audio_small}")

    # For extended benchmark, just pick top 1 largest and smallest (no comparison needed)
    if _extended_benchmark in top_models_per_benchmark:
        _ext_largest = list(top_models_per_benchmark[_extended_benchmark]["largest"])
        _ext_smallest = list(top_models_per_benchmark[_extended_benchmark]["smallest"])
        selected_extended_large = _ext_largest[0] if _ext_largest else None
        selected_extended_small = _ext_smallest[0] if _ext_smallest else None
    else:
        selected_extended_large = None
        selected_extended_small = None

    print(f"\nSelected extended large model: {selected_extended_large}")
    print(f"Selected extended small model: {selected_extended_small}")
    return (
        selected_audio_large,
        selected_audio_small,
        selected_extended_large,
        selected_extended_small,
    )


@app.cell
def _(df_with_size, pd, selected_audio_large, selected_audio_small):
    # Build summary table for MAEB benchmarks (common models)
    _benchmarks = ["MAEB(audio-only)", "MAEB"]

    def _build_audio_row(_mname, _rec, _cat):
        _row = {
            "Size": _cat,
            "Model": _mname.split("/")[-1] if "/" in _mname else _mname,
            "Memory (MB)": round(_rec["memory_usage_mb"], 1)
            if pd.notna(_rec["memory_usage_mb"])
            else "N/A",
            "Params (M)": round(_rec["n_parameters"] / 1e6, 1)
            if pd.notna(_rec["n_parameters"])
            else "N/A",
        }
        for _b in _benchmarks:
            _tcol = f"{_b}_eval_time"
            _ccol = f"{_b}_task_count"
            _etime = _rec.get(_tcol)
            _tc = _rec.get(_ccol, 0)
            # Convert to hours with 3 decimal places
            if pd.notna(_etime):
                _row[f"{_b} (hours)"] = f"{_etime / 3600:.3f}"
                _row[f"{_b} (tasks)"] = int(_tc)
            else:
                _row[f"{_b} (hours)"] = "N/A"
                _row[f"{_b} (tasks)"] = 0
        return _row

    _audio_rows = []
    for _mname, _cat in [
        (selected_audio_large, "Large"),
        (selected_audio_small, "Small"),
    ]:
        if _mname is None:
            continue
        _recs = df_with_size[df_with_size["model_name"] == _mname]
        if len(_recs) > 0:
            _audio_rows.append(_build_audio_row(_mname, _recs.iloc[0], _cat))

    audio_summary_df = pd.DataFrame(_audio_rows) if _audio_rows else pd.DataFrame()
    print("=== MAEB Benchmarks Summary (common models) ===")
    audio_summary_df
    return


@app.cell
def _(df_with_size, pd, selected_extended_large, selected_extended_small):
    # Build summary table for MAEB(extended) benchmark
    _extended_benchmark = "MAEB(extended)"

    def _build_extended_row(_mname, _rec, _cat):
        _row = {
            "Size": _cat,
            "Model": _mname.split("/")[-1] if "/" in _mname else _mname,
            "Memory (MB)": round(_rec["memory_usage_mb"], 1)
            if pd.notna(_rec["memory_usage_mb"])
            else "N/A",
            "Params (M)": round(_rec["n_parameters"] / 1e6, 1)
            if pd.notna(_rec["n_parameters"])
            else "N/A",
        }
        _tcol = f"{_extended_benchmark}_eval_time"
        _ccol = f"{_extended_benchmark}_task_count"
        _etime = _rec.get(_tcol)
        _tc = _rec.get(_ccol, 0)
        # Convert to hours with 3 decimal places
        if pd.notna(_etime):
            _row["Eval Time (hours)"] = f"{_etime / 3600:.3f}"
            _row["Tasks"] = int(_tc)
        else:
            _row["Eval Time (hours)"] = "N/A"
            _row["Tasks"] = 0
        return _row

    _ext_rows = []
    for _mname, _cat in [
        (selected_extended_large, "Large"),
        (selected_extended_small, "Small"),
    ]:
        if _mname is None:
            continue
        _recs = df_with_size[df_with_size["model_name"] == _mname]
        if len(_recs) > 0:
            _ext_rows.append(_build_extended_row(_mname, _recs.iloc[0], _cat))

    extended_summary_df = pd.DataFrame(_ext_rows) if _ext_rows else pd.DataFrame()
    print("=== MAEB(extended) Benchmark Summary ===")
    extended_summary_df
    return


@app.cell
def _(BENCHMARK_NAMES, df_with_size, pd, top_models_per_benchmark):
    # Also show the top 3 largest and smallest per benchmark with their eval times
    all_top_models = []

    for _bname2 in BENCHMARK_NAMES:
        if _bname2 not in top_models_per_benchmark:
            continue

        _tcol3 = f"{_bname2}_eval_time"

        for _category, _model_set in [
            ("largest", top_models_per_benchmark[_bname2]["largest"]),
            ("smallest", top_models_per_benchmark[_bname2]["smallest"]),
        ]:
            for _mname2 in _model_set:
                _recs = df_with_size[df_with_size["model_name"] == _mname2]
                if len(_recs) == 0:
                    continue

                _r = _recs.iloc[0]
                _etime = _r.get(_tcol3)

                all_top_models.append(
                    {
                        "Benchmark": _bname2,
                        "Category": _category,
                        "Model": _mname2.split("/")[-1] if "/" in _mname2 else _mname2,
                        "Memory (MB)": round(_r["memory_usage_mb"], 1)
                        if pd.notna(_r["memory_usage_mb"])
                        else None,
                        "Params (M)": round(_r["n_parameters"] / 1e6, 1)
                        if pd.notna(_r["n_parameters"])
                        else None,
                        "Eval Time (hours)": f"{_etime / 3600:.3f}"
                        if pd.notna(_etime)
                        else None,
                    }
                )

    detailed_df = pd.DataFrame(all_top_models)
    detailed_df
    return


@app.cell
def _():
    import marimo as mo

    return


if __name__ == "__main__":
    app.run()
