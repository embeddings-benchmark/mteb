"""Filters that remove low-quality samples from a task before it is evaluated.

Each filter takes a task and returns a cleaned copy:

```python
import mteb
from mteb.data_cleaning import remove_duplicates, remove_short_texts

task = remove_duplicates(mteb.get_task("MassiveIntentClassification"))
task = remove_short_texts(task, min_length=1)
```

"""

from __future__ import annotations

from ._filters import (
    remove_duplicates,
    remove_short_audio,
    remove_short_texts,
    remove_short_videos,
    remove_small_images,
)

__all__ = [
    "remove_duplicates",
    "remove_short_audio",
    "remove_short_texts",
    "remove_short_videos",
    "remove_small_images",
]
