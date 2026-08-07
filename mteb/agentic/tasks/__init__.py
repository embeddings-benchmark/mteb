"""Answer-mode task registry, mirroring mteb.get_tasks.

One module per task. To add a task, create a module defining a TaskMeta and
list it in TASK_REGISTRY.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mteb.agentic.tasks.browsecomp_plus import browsecomp_plus
from mteb.agentic.tasks.hotpotqa import hotpotqa
from mteb.agentic.tasks.longbench_v2 import longbench_v2
from mteb.agentic.tasks.multihop_rag import multihop_rag
from mteb.agentic.tasks.musique import musique
from mteb.agentic.tasks.oolong import oolong

if TYPE_CHECKING:
    from mteb.agentic.data import AnswerTaskData, TaskMeta

TASK_REGISTRY: dict[str, TaskMeta] = {
    meta.name: meta
    for meta in [
        browsecomp_plus,
        hotpotqa,
        musique,
        multihop_rag,
        oolong,
        longbench_v2,
    ]
}


def list_tasks() -> list[str]:
    """Names of all registered answer-mode tasks."""
    return sorted(TASK_REGISTRY)


def get_task_meta(name: str) -> TaskMeta:
    """Fetch a task's metadata by name."""
    if name not in TASK_REGISTRY:
        from difflib import get_close_matches

        suggestion = get_close_matches(name, TASK_REGISTRY, n=1)
        hint = f" Did you mean {suggestion[0]!r}?" if suggestion else ""
        raise KeyError(f"Unknown task {name!r}.{hint} Available: {list_tasks()}")
    return TASK_REGISTRY[name]


def get_task(name: str, **kwargs: Any) -> AnswerTaskData:
    """Load a registered answer-mode task by name."""
    return get_task_meta(name).load(**kwargs)


__all__ = ["TASK_REGISTRY", "get_task", "get_task_meta", "list_tasks"]
