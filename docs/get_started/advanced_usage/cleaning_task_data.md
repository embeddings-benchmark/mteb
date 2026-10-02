---
title: "Cleaning task data"
icon: lucide/brush-cleaning
---

# Cleaning Task Data

Some datasets have quality issues. A dataset may repeat the same document many times, or contain documents that are empty or too short to carry meaning. Both distort a benchmark: a duplicated document is scored twice, and an empty one is scored on nothing. The same goes for images, audio and video, e.g. a 1x1 pixel placeholder image or an audio clip cut to a fraction of a second.

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

Fewer unique texts than samples means the split contains duplicates -- four of them here. A small `min_text_length` points the other way, at documents too short to be meaningful. For images, audio and video, look at `min_image_width`, `min_image_height` and `min_duration_seconds` instead.

For a task you are developing, compute the statistics yourself with [`task.calculate_descriptive_statistics()`][mteb.AbsTask.calculate_descriptive_statistics].

## Available filters

Each filter takes a task and returns a cleaned copy, leaving the task you passed in untouched. They cover every split and subset by default; see the linked reference for the arguments that narrow that down.

- [`remove_duplicates`][mteb.data_cleaning.remove_duplicates] removes [repeated samples](#removing-duplicates).
- [`remove_short_texts`][mteb.data_cleaning.remove_short_texts] removes [empty and too-short texts](#removing-short-or-small-samples).
- [`remove_small_images`][mteb.data_cleaning.remove_small_images], [`remove_short_audio`][mteb.data_cleaning.remove_short_audio] and [`remove_short_videos`][mteb.data_cleaning.remove_short_videos] do the same for [images, audio and video](#images-audio-and-video).

Filters can be chained, e.g. `remove_short_texts(remove_duplicates(task), min_length=1)`.

## Removing duplicates

[`remove_duplicates`][mteb.data_cleaning.remove_duplicates] drops repeated samples, keeping the first of each:

```python
from mteb.data_cleaning import remove_duplicates

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

Text is compared exactly as written, and Unicode can spell an accented letter either as one code point or as a letter followed by a combining mark. Composing it with NFC lets duplicates that differ only in that match, which this task's Vietnamese split needs, as a fifth of its texts are decomposed:

```python
import unicodedata

cleaned = remove_duplicates(
    task, normalization=lambda text: unicodedata.normalize("NFC", text.strip())
)
# finds 98 duplicates in the Vietnamese train split that the default comparison misses
```

Only text is normalized; images, audio and video are compared by an exact hash of their content, so the filter works on any task but does not match a re-encoded or rescaled copy of a sample. Retrieval tasks keep their relevance judgements valid: a judgement pointing at a removed duplicate moves to the copy that was kept. Their documents and queries are compared as the model reads them, so a document's title is part of its text, and two queries that differ only in their instruction are not duplicates.

## Removing short or small samples

[`remove_short_texts`][mteb.data_cleaning.remove_short_texts] drops samples whose text is shorter than `min_length`:

```python
from mteb.data_cleaning import remove_short_texts

cleaned = remove_short_texts(task, min_length=3)

print({split: len(data) for split, data in cleaned.dataset["en"].items()})
# {'train': 11511, 'test': 2973, 'validation': 2033}, from 11514 / 2974 / 2033
```

Texts are counted in characters without their surrounding whitespace, so `min_length=1` removes exactly the empty and whitespace-only ones. Pass `length_fn` to measure them differently, e.g. `lambda text: len(text.split())` to count words.

There is no default threshold, as what is too short depends on the task: the one-word texts of this one include `"s."`, but also terse yet genuine commands such as `"coffee"` and `"remind"`.

### Thresholds depend on the writing system

A character carries more meaning in some scripts than in others, so a threshold does not travel between languages. In MassiveIntentClassification the median test text is 32 characters in English but 10 in Chinese, and `min_length=10` removes 1.8% of the English subset against 48.5% of the Chinese one. Counting words is worse: Chinese, Japanese and Thai leave no spaces between words, so a threshold of three words removes 99% of that Chinese subset.

Filter a multilingual task one language at a time; the calls chain into a single cleaned task:

```python
cleaned = remove_short_texts(task, min_length=10, subsets=["en"])
cleaned = remove_short_texts(cleaned, min_length=2, subsets=["zh-CN"])
```

Two details follow from counting code points rather than characters as a reader sees them. A zero-width space is not whitespace, so it survives `min_length=1`. And a vowel sign or virama counts on its own, so this task's Tamil texts measure 1.6 times, and its Thai texts 1.3 times, their length in characters -- which makes a threshold that much more permissive for them. Pass `length_fn` if either matters, e.g. `lambda text: len(regex.findall(r"\X", text))` to count characters.

### Images, audio and video

The other modalities have a filter of their own, each measuring its content the way the descriptive statistics do:

```python
from mteb.data_cleaning import (
    remove_short_audio,
    remove_short_videos,
    remove_small_images,
)

cleaned = remove_small_images(task, min_size=32)  # width or height under 32 pixels
cleaned = remove_short_audio(task, min_seconds=0.5)
cleaned = remove_short_videos(task, min_seconds=1.0)
```

An image counts as small by its shorter side, so `min_size=32` asks for 32 pixels in both directions; pass `size_fn` to measure it differently, e.g. `lambda image: image.width * image.height` for its area. Their thresholds need the same care as a text's: a 28x28 MNIST digit and a 0.1 second drum hit are small by nature, not broken.

### How samples are measured

A sample is removed when any of its values is too small, e.g. either sentence of a pair. Each filter only measures its own modality, so `remove_short_texts` keeps the images of a task whatever their size. In a retrieval task a document is measured on its title and text together, and a query together with its instruction, as that is what the model reads. A document or query that combines modalities, such as a page image with its extracted text, is left alone, as one may carry what the other lacks. The relevance judgements of a removed document are dropped with it, and a query left without any relevant document is dropped as well, as it can no longer be scored.

## Cleaning produces a new task

A cleaned task is a different task, so it is given an id of its own rather than reusing the published one:

```python
cleaned = remove_duplicates(task)

print(task.metadata.name)  # MassiveIntentClassification
print(cleaned.metadata.name)  # MassiveIntentClassification (remove_duplicates)
```

Each filter adds its name to the list, so applying a second one gives `MassiveIntentClassification (remove_duplicates, remove_short_texts)`. The task you passed in keeps its own name, and `adapted_from` on the copy records where the data came from.

That id is what keeps the result honest. You evaluate a cleaned task as usual, and its scores are recorded against the cleaned id rather than against the published dataset:

```python
results = mteb.evaluate(model, [cleaned])
print(results[0].task_name)  # MassiveIntentClassification (remove_duplicates)
```

!!! note
    As cleaning the dataset changes the score, we do not accept scores from modified datasets on the
    [leaderboard](https://huggingface.co/spaces/mteb/leaderboard). We do however allow a cleaned version of a
    dataset to be [submitted to MTEB](../../contributing/adding_a_dataset.md#improving-or-cleaning-a-task).
