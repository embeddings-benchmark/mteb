"""Filters that remove low-quality samples from a task before it is evaluated.

Each filter takes a task and returns a cleaned copy:

```python
import mteb
from mteb.data_cleaning import remove_by_text_length, remove_duplicates

task = remove_duplicates(mteb.get_task("MassiveIntentClassification"))
task = remove_by_text_length(task, min_length=1)
```

"""

from __future__ import annotations

from ._content_size import (
    Quantile,
    remove_by_audio_duration,
    remove_by_image_size,
    remove_by_text_length,
    remove_by_video_duration,
)
from ._duplicates import remove_duplicates

__all__ = [
    "Quantile",
    "remove_by_audio_duration",
    "remove_by_image_size",
    "remove_by_text_length",
    "remove_by_video_duration",
    "remove_duplicates",
]
