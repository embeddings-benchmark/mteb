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

# Default cache location (~/.cache/mteb)
cache = mteb.ResultCache()

# Custom location
cache = mteb.ResultCache(cache_dir="/path/to/cache")
```

```python
# This clones/fetches the latest results from remote repository, creates the directory structure, and prepares for submissions.
cache.download_from_remote()
```

### Environment Variables

```bash
# Set custom cache location
export MTEB_CACHE=/path/to/cache

# To clear cache if needed remove the directory or use:
cache.clear_cache()
```

## Quick Start

Complete example: evaluate, cache, and submit results:

```python
import mteb

# 1. Initialize cache
cache = mteb.ResultCache()
cache.download_from_remote()

# 2. Evaluate model
model = mteb.get_model("sentence-transformers/all-MiniLM-L6-v2")
model_meta = mteb.get_model_meta("sentence-transformers/all-MiniLM-L6-v2")
tasks = mteb.get_tasks(["STS12","STS13"])

results = mteb.evaluate(model, tasks=tasks, cache=cache)

# 3. Save to cache
for task_result in results.task_results:
    cache.save_to_cache(
        task_result=task_result,
        model_name=model_meta.name,
        model_revision=model_meta.revision
    )

# 4. Submit results (manual review before pushing)
cache.submit_results(models=["sentence-transformers/all-MiniLM-L6-v2"], create_pr=False)
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

# Access results
for model_name, model_results in results.items():
    for task_name, task_result in model_results.items():
        print(f"{model_name} on {task_name}: {task_result.scores}")
```

## Submitting Results

### Manual Submission (Recommended)

Prepare results without automatically creating a PR:

```python
submission_info = cache.submit_results(
    models=["sentence-transformers/all-MiniLM-L6-v2"],
    create_pr=False
)

# Review instructions
print(submission_info.get("manual_submission_instructions"))
# Manually review and push when ready
```

### Automated Submission

Automatically create a pull request. GitHub integration must be configured first, either by signing in with `gh auth login` or setting up your Git credential helper.

Then run your code:

```python
submission_info = cache.submit_results(
    models=["sentence-transformers/all-MiniLM-L6-v2"],
    create_pr=True
)

if submission_info.get("pr_url"):
    print(f"PR created: {submission_info['pr_url']}")
```

ResultCache will automatically use the configured GitHub integration.

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
