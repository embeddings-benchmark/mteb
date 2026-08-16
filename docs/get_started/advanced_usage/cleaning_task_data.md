---
title: "Cleaning task data"
icon: lucide/brush-cleaning
---

# Cleaning Task Data

Some datasets have quality issues. A dataset may repeat the same document many times, or contain documents that are
empty or too short to carry meaning. Both distort a benchmark: a duplicated document is scored twice, and an empty one is scored on nothing.

## Spotting the issue

The descriptive statistics of a task are the quickest way to see this. They are published with the task, so you can
inspect them without downloading the data:

```python
import mteb

task = mteb.get_task("MassiveIntentClassification", languages=["eng"])
stats = task.metadata.descriptive_stats["test"]["hf_subset_descriptive_stats"]["en"]

print(stats["num_samples"])  # 2974
print(stats["text_statistics"]["unique_texts"])  # 2970
print(stats["text_statistics"]["min_text_length"])  # 2
```

Fewer unique texts than samples means the split contains duplicates -- four of them here. A small `min_text_length` points the other way, at documents too short to be meaningful.

For a task you are developing, compute the statistics yourself with
[`task.calculate_descriptive_statistics()`][mteb.AbsTask.calculate_descriptive_statistics].

## Available filters

Each filter takes a task, modifies its dataset in place and returns it, so they can be chained. They cover every split and subset by default; see the linked reference for the arguments.

| filter | removes |
|---|---|
| [`remove_duplicates`][mteb.quality.remove_duplicates] | [repeated samples](#removing-duplicates) |

## Removing duplicates

[`remove_duplicates`][mteb.quality.remove_duplicates] drops repeated samples, keeping the first of each:

```python
from mteb.quality import remove_duplicates

task = remove_duplicates(task)

print({split: len(data) for split, data in task.dataset["en"].items()})
# {'train': 11468, 'test': 2970, 'validation': 2031}, from 11514 / 2974 / 2033
```

Two documents count as duplicates when they are identical once surrounding whitespace is stripped. `normalize=`
loosens that to ignore case, or case and punctuation:

```python
task = remove_duplicates(
    task, normalize="alphanumeric"
)  # "Wake me up!" == "wake  me  up"
```

Text is compared as text, while images, audio and video are compared by a hash of their content, so the filter
works on any task. Retrieval tasks keep their relevance judgements valid: a judgement pointing at a removed
duplicate moves to the copy that was kept.

## Changes after cleaning Task Data

Cleaning changes the data, and it would change the scores too, because the model would be
evaluated on different data. What it does *not* change is the task's name or its `dataset_revision`, so such a
score would be indistinguishable from one computed on the published dataset while not being comparable to it.

For that reason a cleaned task is marked with `data_modified` and [`mteb.evaluate`][mteb.evaluate] refuses to run
it:

```python
mteb.evaluate(model, task)
# ValueError: The data of ['MassiveIntentClassification'] was modified locally ...
```


!!! warning
    Scores from a locally cleaned task are not accepted on the
    [leaderboard](https://huggingface.co/spaces/mteb/leaderboard). They are not comparable to any other result, and
    nothing in the result file would distinguish them.

## Contributing the fix

If a dataset needs cleaning, everyone benefits from fixing it once rather than in each user's script. Submit the
cleaned data as a **new version of the task** rather than as scores. A new version:

- bumps the version suffix of the task name, so `MassiveIntentClassification` becomes
  `MassiveIntentClassification.v2`. The suffix replaces rather than accumulates, so a task that is already at `.v2` becomes `.v3`.
- points `adapted_from` at the task it was derived from, and sets `superseded_by` on every older version so that
  users of those are warned towards the current one.

See [Adding a Task](../../contributing/adding_a_dataset.md) for the full process, and
[`push_dataset_to_hub`][mteb.AbsTask.push_dataset_to_hub] for uploading the cleaned data.
`scripts/data/clean_and_update_tasks.py` automates the version bump, including finding the highest existing
version.

That way the fix keeps its own identity, and results on the old and the new version remain distinguishable.
