"""Filters that remove low-quality samples from a task before it is evaluated.

Each filter takes a task, modifies its dataset in place and returns it:

```python
import mteb
from mteb.quality import remove_duplicates

task = remove_duplicates(mteb.get_task("MassiveIntentClassification"))
```

"""

from __future__ import annotations

from ._filters import remove_duplicates

__all__ = [
    "remove_duplicates",
]
