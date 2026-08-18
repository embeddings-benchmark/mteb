"""Filters that remove low-quality samples from a task before it is evaluated.

Each filter takes a task, modifies its dataset in place and returns it:

```python
import mteb
from mteb.quality import remove_duplicates

task = remove_duplicates(mteb.get_task("MassiveIntentClassification"))
```

Only deduplication is exposed for now. `mteb/abstasks/_data_filter/` holds further filters -- train/test leakage,
contradictory labels, and creating a test split -- that are implemented but not yet reachable from here; see
https://github.com/embeddings-benchmark/mteb/issues/3672 for which of them to expose and how.
"""

from __future__ import annotations

from ._filters import (
    alphanumeric_text,
    casefold_text,
    remove_duplicates,
    strip_whitespace,
)

__all__ = [
    "alphanumeric_text",
    "casefold_text",
    "remove_duplicates",
    "strip_whitespace",
]
