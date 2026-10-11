---
title: "Submit results"
icon: lucide/upload
---

# Submit Results

## Overview

The [`ResultCache`][mteb.cache.result_cache.ResultCache] class manages evaluation results locally and submits them to the [official results repository](https://github.com/embeddings-benchmark/results). Use it to cache results, avoid re-computation, and contribute results back to the community.

If you want to request that a model be evaluated, rather than submitting results that you ran yourself, open an [Evaluation Request](https://github.com/embeddings-benchmark/mteb/issues/new?template=eval_request.yaml) issue.

## Loading Results

For a full guide on loading and working with results — including filtering, dataframe conversion, and benchmark scoring — see [Loading Results](../get_started/usage/loading_results.md).

## Quick Start

Complete example: evaluate, cache, and submit results:

=== "Python"

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

=== "CLI"

    ```bash
    # 1. Evaluate model
    mteb run -m sentence-transformers/all-MiniLM-L6-v2 -t ArguAna

    # 2. Submit results
    mteb submit-results -m sentence-transformers/all-MiniLM-L6-v2  # manual review before pushing
    ```

    To evaluate and open a PR in a single command, add `--create-pr` to `mteb run`. Only the results of the evaluated tasks are submitted:

    ```bash
    mteb run -m sentence-transformers/all-MiniLM-L6-v2 -t ArguAna --create-pr
    ```

Only results that the results repository does not have yet are submitted. Results computed on a different dataset revision than those in the repository are skipped with a warning.

## Submitting Results
=== "Manual Submission"
    !!! note Requirements
        Git is required for this action.

    Prepare results without automatically creating a PR:

    === "Python"

        ```python
        submission_info = cache.submit_results(
            models=["sentence-transformers/all-MiniLM-L6-v2"], create_pr=False
        )

        # submit_results logs the manual submission instructions
        print(f"Prepared submission at: {submission_info['path']}")
        ```

    === "CLI"

        ```bash
        # logs the manual submission instructions
        mteb submit-results -m sentence-transformers/all-MiniLM-L6-v2
        ```

=== "Automated Submission"
    !!! note Requirements
        Git, [GitHub CLI](https://cli.github.com) are required for this action. You also need to install the `mteb[github]` extra dependencies and configure GitHub integration by signing in with `gh auth login` or setting up your Git credential helper.

        === "pip"
            ```bash
            pip install mteb[github]
            ```
        === "uv"
            ```bash
            uv pip install mteb[github]
            ```
    Then run your code:

    === "Python"

        ```python
        submission_info = cache.submit_results(
            models=["sentence-transformers/all-MiniLM-L6-v2"],
            create_pr=True,
            message="Add ArguAna results for all-MiniLM-L6-v2",
        )

        if submission_info.get("pr_url"):
            print(f"PR created: {submission_info['pr_url']}")
        ```

    === "CLI"

        ```bash
        mteb submit-results -m sentence-transformers/all-MiniLM-L6-v2 \
            --create-pr \
            --message "Add ArguAna results for all-MiniLM-L6-v2"
        ```

    The message is used as the title of the commit and PR; if omitted, a title listing the submitted models is used.

### Batch Submission

Submit multiple models at once:

=== "Python"

    ```python
    models = [
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/all-mpnet-base-v2",
        "BAAI/bge-base-en-v1.5",
    ]

    cache.submit_results(models=models, create_pr=False)
    ```

=== "CLI"

    ```bash
    mteb submit-results -m sentence-transformers/all-MiniLM-L6-v2 \
        sentence-transformers/all-mpnet-base-v2 \
        BAAI/bge-base-en-v1.5

    # without -m, the results of all models are submitted
    mteb submit-results
    ```

You can also restrict the submission to specific tasks with `tasks=[...]` in Python or `-t` in the CLI.


### After submission

Once the PR is created your result will now wait for review, we aim for this to take less than a week. To speed up the review please make sure the fill
out the checklist. During the review process we might ask you about suspicious results or ask you to check for potential data leakage.

## API Reference

- [`submit_results()`][mteb.cache.result_cache.ResultCache.submit_results] - Submit results
- [`save_to_cache()`][mteb.cache.result_cache.ResultCache.save_to_cache] - Save evaluation results
- [`load_results()`][mteb.cache.result_cache.ResultCache.load_results] - Load cached results
- [`download_from_remote()`][mteb.cache.result_cache.ResultCache.download_from_remote] - Sync with remote
- [`clear_cache()`][mteb.cache.result_cache.ResultCache.clear_cache] - Clear cache
