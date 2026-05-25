"""Compute data contamination (training overlap) for all models with MVEB results."""

import logging

import mteb
from mteb.models.get_model_meta import get_model_meta

logging.disable(logging.WARNING)

# Get MVEB benchmark tasks
benchmark = mteb.get_benchmark("MVEB")
task_names = set(t.metadata.name for t in benchmark.tasks)

import pathlib
results_path = pathlib.Path("/Users/adnan/research/mveb/results/results")

models_with_results = []
for model_dir in sorted(results_path.iterdir()):
    if not model_dir.is_dir():
        continue
    found_tasks = set()
    for rev_dir in model_dir.iterdir():
        if not rev_dir.is_dir():
            continue
        for f in rev_dir.iterdir():
            if f.stem in task_names:
                found_tasks.add(f.stem)
    if found_tasks:
        model_name = model_dir.name.replace("__", "/")
        models_with_results.append((model_name, found_tasks))

print(f"Models with MVEB results: {len(models_with_results)}")
print(f"MVEB tasks: {len(task_names)}")
print()
print(f"{'Model':<50} {'Zero-shot %':>11} {'# Overlap':>9} {'Overlapping Tasks'}")
print("-" * 100)

contaminated = []
zero_shot = []

for model_name, result_tasks in models_with_results:
    try:
        meta = get_model_meta(model_name)
    except KeyError:
        print(f"{model_name:<50} {'NOT IN REGISTRY':>11}")
        continue

    training_data = meta.training_datasets or set()
    overlap = task_names & (
        training_data if isinstance(training_data, set) else set(training_data)
    )
    n_tasks = len(result_tasks)
    n_overlap = len(overlap)
    zero_shot_pct = ((n_tasks - n_overlap) / n_tasks * 100) if n_tasks > 0 else 100
    overlap_str = ", ".join(sorted(overlap)) if overlap else ""

    row = (model_name, zero_shot_pct, n_overlap, overlap_str, n_tasks)
    if n_overlap > 0:
        contaminated.append(row)
    else:
        zero_shot.append(row)

    print(f"{model_name:<50} {zero_shot_pct:>10.0f}% {n_overlap:>9} {overlap_str}")

print()
print(f"Summary: {len(zero_shot)} zero-shot, {len(contaminated)} contaminated")
