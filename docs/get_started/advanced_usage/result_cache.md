---
title: "Result Caching and Submission"
icon: lucide/save
---

## Overview

The [`ResultCache`][mteb.cache.result_cache.ResultCache] class manages evaluation results locally and submits them to the [official results repository](https://github.com/embeddings-benchmark/results). Use it to cache results, avoid re-computation, and contribute results back to the community.

## Setup

### Create and Initialize Cache

```python
import mteb

cache = mteb.ResultCache(cache_path="~/.cache/mteb") # (1)! 
cache.download_from_remote() # (2)!
```

1. Also, possible to set via `MTEB_CACHE` environment variable
2. Optional: clone/fetch the latest remote results before loading remote results or submitting.

## Quick Start

Complete example: evaluate, cache, and submit results:

```python
import mteb

# 1. Initialize cache
cache = mteb.ResultCache()

# 2. Evaluate model
model_meta = mteb.get_model_meta("sentence-transformers/all-MiniLM-L6-v2")
task = mteb.get_task("ArguAna")

mteb.evaluate(model_meta, task, cache=cache)

# 3. Submit results
cache.submit_results(model_meta, create_pr=False)  # manual review before pushing
```

## Loading Results

```python
# Load results for specific models
results = cache.load_results(models=["sentence-transformers/all-MiniLM-L6-v2"])

# Filter by tasks and language
results = cache.load_results(
    models=["sentence-transformers/all-MiniLM-L6-v2"],
    tasks=["STS12", "STS13"],
    langs=["en"]
)

for model_name, model_results in results.items():
    for task_name, task_result in model_results.items():
        print(f"{model_name} on {task_name}: {task_result.get_score()}")
```

## Submitting Results
=== "Manual Submission"
    !!! warning Requirements
        Git is required for this action.

    Prepare results without automatically creating a PR:
    
    ```python
    submission_info = cache.submit_results(
        models=["sentence-transformers/all-MiniLM-L6-v2"],
        create_pr=False
    )
    
    # Review instructions
    print(submission_info["manual_submission_instructions"])
    ```

=== "Automated Submission"
    !!! warning Requirements
        Git, [Github CLI](https://cli.github.com) are required for this action. You also need to install the `mteb[github]` extra dependencies and configure GitHub integration by signing in with `gh auth login` or setting up your Git credential helper.

        === "pip"
            ```bash
            pip install mteb[github]
            ```
        === "uv"
            ```bash
            uv pip install mteb[github]
            ```
    Then run your code:
    
    ```python
    submission_info = cache.submit_results(
        models=["sentence-transformers/all-MiniLM-L6-v2"],
        create_pr=True
    )
    
    if submission_info.get("pr_url"):
        print(f"PR created: {submission_info['pr_url']}")
    ```

### Batch Submission

Submit multiple models at once:

```python
models = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "BAAI/bge-base-en-v1.5"
]

cache.submit_results(models=models, create_pr=False)
```

## API Reference

- [`submit_results()`][mteb.cache.result_cache.ResultCache.submit_results] - Submit results
- [`save_to_cache()`][mteb.cache.result_cache.ResultCache.save_to_cache] - Save evaluation results
- [`load_results()`][mteb.cache.result_cache.ResultCache.load_results] - Load cached results
- [`download_from_remote()`][mteb.cache.result_cache.ResultCache.download_from_remote] - Sync with remote
- [`clear_cache()`][mteb.cache.result_cache.ResultCache.clear_cache] - Clear cache
