from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from numpy.typing import NDArray

BT_SCORE_BASE = 1000.0
BT_SCORE_SCALE = 400.0 / math.log(10.0)
# Virtual draws added to every ordered pair so undefeated / winless models stay finite.
_PSEUDO_WINS = 0.5
_MAX_ITER = 200
_TOL = 1e-9
# Bootstrap chunking keeps the (chunk, M, M) weighted win tensor small.
_BOOT_CHUNK = 16


@dataclass(frozen=True, slots=True)
class BtScoreResult:
    """Rating for one row, with an optional 95% bootstrap interval."""

    score: float
    low: float | None = None
    high: float | None = None


def _task_win_matrices(scores: NDArray[np.float64]) -> NDArray[np.float32]:
    """Per-task pairwise win matrices, shape ``(T, M, M)`` (float32).

    ``scores`` is ``(M, T)`` with NaN for missing. ``out[t, i, j]`` is 1 if model i
    beats j on task t, 0.5 on a tie, 0 otherwise; pairs where both lack a score are 0.
    """
    n_models, n_tasks = scores.shape
    out = np.zeros((n_tasks, n_models, n_models), dtype=np.float32)
    for t in range(n_tasks):
        col = scores[:, t]
        present = ~np.isnan(col)
        filled = np.where(present, col, -np.inf)
        both_missing = ~present[:, None] & ~present[None, :]
        wins = (filled[:, None] > filled[None, :]).astype(np.float32)
        ties = (filled[:, None] == filled[None, :]) & ~both_missing
        wins += 0.5 * ties
        np.fill_diagonal(wins, 0.0)
        out[t] = wins
    return out


def _fit(
    wins: NDArray[np.floating], init: NDArray[np.float64] | None = None
) -> NDArray[np.float64]:
    """Bradley-Terry log-strengths (mean-centered) from a ``(M, M)`` win matrix."""
    n_models = wins.shape[0]
    w = wins.astype(np.float64) + _PSEUDO_WINS * (1.0 - np.eye(n_models))
    n_ij = w + w.T
    total_wins = w.sum(axis=1)
    p = np.ones(n_models) if init is None else np.exp(init)
    for _ in range(_MAX_ITER):
        denom = (n_ij / (p[:, None] + p[None, :])).sum(axis=1)
        new_p = total_wins / denom
        new_p /= np.exp(np.log(new_p).mean())
        if np.max(np.abs(np.log(new_p) - np.log(p))) < _TOL:
            p = new_p
            break
        p = new_p
    log_p = np.log(p)
    centered: NDArray[np.float64] = log_p - log_p.mean()
    return centered


def _to_score(log_strength: NDArray[np.float64]) -> NDArray[np.float64]:
    return BT_SCORE_BASE + BT_SCORE_SCALE * log_strength


def _bootstrap_interval(
    per_task: NDArray[np.float32], base_log: NDArray[np.float64], n_boot: int, seed: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """95% interval of each row's BT score over task resamples (with replacement)."""
    n_tasks, n_models, _ = per_task.shape
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(n_tasks, np.full(n_tasks, 1.0 / n_tasks), size=n_boot)
    draws = draws.astype(np.float32)
    flat = per_task.reshape(n_tasks, -1)
    boot = np.empty((n_boot, n_models))
    for start in range(0, n_boot, _BOOT_CHUNK):
        chunk = draws[start : start + _BOOT_CHUNK]
        wins_b = (chunk @ flat).reshape(len(chunk), n_models, n_models)
        for b in range(len(chunk)):
            boot[start + b] = _to_score(_fit(wins_b[b], init=base_log))
    low, high = np.percentile(boot, [2.5, 97.5], axis=0)
    return low, high


def compute_bt_score(
    scores: Mapping[tuple[str, str], Mapping[str, float]],
    tasks: Sequence[str],
    *,
    n_boot: int = 100,
    seed: int = 0,
) -> dict[tuple[str, str], BtScoreResult]:
    """Rate every row of ``scores`` over ``tasks``.

    Args:
        scores: ``(model name, experiment id) -> {task: score}``; a task absent from the mapping (or NaN) is missing.
        tasks: Tasks to compare on; scores for other tasks are ignored.
        n_boot: Bootstrap resamples over tasks for the 95% interval; ``0`` disables it.
        seed: RNG seed so the output is deterministic (and cacheable).

    Returns:
        ``(model name, experiment id) -> BtScoreResult``. Empty if there are fewer than two rows or no tasks.
    """
    keys = list(scores)
    tasks = list(dict.fromkeys(tasks))
    if len(keys) < 2 or not tasks:
        return {}

    matrix = np.full((len(keys), len(tasks)), np.nan)
    for i, key in enumerate(keys):
        row = scores[key]
        for t, task in enumerate(tasks):
            v = row.get(task)
            if v is not None and not math.isnan(v):
                matrix[i, t] = v

    per_task = _task_win_matrices(matrix)
    base_log = _fit(per_task.sum(axis=0))
    score = _to_score(base_log)

    low = high = None
    if n_boot > 0 and len(tasks) > 1:
        low, high = _bootstrap_interval(per_task, base_log, n_boot, seed)

    return {
        key: BtScoreResult(
            score=float(score[i]),
            low=None if low is None else float(min(low[i], score[i])),
            high=None if high is None else float(max(high[i], score[i])),
        )
        for i, key in enumerate(keys)
    }
