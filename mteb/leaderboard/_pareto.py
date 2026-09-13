"""Pareto comparisons for leaderboard models."""

from __future__ import annotations

from math import isfinite
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


def _pareto_frontier(
    scores: Sequence[float | None],
    sizes: Sequence[float | None],
) -> list[bool | None]:
    """Maximize score and minimize size, preserving input order.

    Invalid observations receive None. Exact ties do not dominate each other.
    """
    if len(scores) != len(sizes):
        raise ValueError("Scores and sizes must have the same length.")

    result: list[bool | None] = [None] * len(scores)
    candidates: list[tuple[float, float, int]] = []

    for index, (score, size) in enumerate(zip(scores, sizes, strict=True)):
        if (
            score is None
            or size is None
            or not isfinite(score)
            or not isfinite(size)
            or size < 0
        ):
            continue
        candidates.append((size, score, index))

    candidates.sort(key=lambda item: (item[0], -item[1]))

    best_score = float("-inf")
    best_size = float("inf")

    for size, score, index in candidates:
        if score > best_score:
            result[index] = True
            best_score = score
            best_size = size
        elif score == best_score and size == best_size:
            result[index] = True
        else:
            result[index] = False

    return result
