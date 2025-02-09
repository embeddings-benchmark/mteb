from __future__ import annotations

from itertools import islice


# https://docs.python.org/3/library/itertools.html#itertools.batched
# Added in version 3.12.
def batched(iterable, n: int, *, strict: bool = False) -> tuple:
    # batched('ABCDEFG', 3) → ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    iterator = iter(iterable)
    while batch := tuple(islice(iterator, n)):
        if strict and len(batch) != n:
            raise ValueError("batched(): incomplete batch")
        yield batch
