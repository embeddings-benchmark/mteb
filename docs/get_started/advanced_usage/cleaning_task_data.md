---
title: "Cleaning task data"
icon: lucide/brush-cleaning
---

# Cleaning Task Data

Some datasets have quality issues. A dataset may repeat the same document many times, or contain documents that are empty or too short to carry meaning. Both distort a benchmark: a duplicated document is scored twice, and an empty one is scored on nothing.

## Spotting the issue

The descriptive statistics of a task are the quickest way to see this. They are published with the task, so you can inspect them without downloading the data:

```python
import mteb

task = mteb.get_task("MassiveIntentClassification", languages=["eng"])
stats = task.metadata.descriptive_stats["test"]["hf_subset_descriptive_stats"]["en"]

print(stats["num_samples"])  # 2974
print(stats["text_statistics"]["unique_texts"])  # 2970
print(stats["text_statistics"]["min_text_length"])  # 2
```

Fewer unique texts than samples means the split contains duplicates -- four of them here. A small `min_text_length` points the other way, at documents too short to be meaningful.

For a task you are developing, compute the statistics yourself with [`task.calculate_descriptive_statistics()`][mteb.AbsTask.calculate_descriptive_statistics].

## Available filters

Each filter takes a task and returns a cleaned copy, leaving the task you passed in untouched. They cover every split and subset by default; see the linked reference for the arguments that narrow that down.

- [`remove_duplicates`][mteb.quality.remove_duplicates] removes [repeated samples](#removing-duplicates).


## Removing duplicates

[`remove_duplicates`][mteb.quality.remove_duplicates] drops repeated samples, keeping the first of each:

```python
from mteb.quality import remove_duplicates

cleaned = remove_duplicates(task)

print({split: len(data) for split, data in cleaned.dataset["en"].items()})
# {'train': 11468, 'test': 2970, 'validation': 2031}, from 11514 / 2974 / 2033
```

Two texts are duplicates when `normalization` rewrites both to the same string. It defaults to `str.strip`, so
only surrounding whitespace is ignored. Pass any function of your own to loosen that:

```python
import re


def casefold_text(text: str) -> str:
    """Also ignore case, so that "Wake me up!" and "wake me up!" match."""
    return text.strip().casefold()


def alphanumeric_text(text: str) -> str:
    """Also ignore punctuation and repeated whitespace, so that "e-mail" and "email" match."""
    return " ".join(re.sub(r"[^\w\s]", "", text.casefold()).split())


cleaned = remove_duplicates(task, normalization=casefold_text)
```

Only text is normalized; images, audio and video are compared by an exact hash of their content, so the filter works on any task but does not match a re-encoded or rescaled copy of a sample. Retrieval tasks keep their relevance judgements valid: a judgement pointing at a removed duplicate moves to the copy that was kept.

## Cleaning produces a new task

A cleaned task is a different task, so it is given an id of its own rather than reusing the published one:

```python
cleaned = remove_duplicates(task)

print(task.metadata.name)  # MassiveIntentClassification
print(cleaned.metadata.name)  # MassiveIntentClassification (remove_duplicates)
```

Each filter adds its name to the list, so applying a second one gives `MassiveIntentClassification (remove_duplicates, filter_short)`. The task you passed in keeps its own name, and
`adapted_from` on the copy records where the data came from.

That id is what keeps the result honest. You evaluate a cleaned task as usual, and its scores are recorded against the cleaned id rather than against the published dataset:

```python
results = mteb.evaluate(model, [cleaned])
print(results[0].task_name)  # MassiveIntentClassification (remove_duplicates)
```

!!! note
    As cleaning the dataset changes the score, we do not accept scores from modified datasets on the
    [leaderboard](https://huggingface.co/spaces/mteb/leaderboard). We do however allow a cleaned version of a
    dataset to be [submitted to MTEB](../../contributing/adding_a_dataset.md#improving-or-cleaning-a-task).
