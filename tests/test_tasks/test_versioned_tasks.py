"""Tests to ensure versioned tasks (.vN) have version_reason set."""

import re

import mteb


def test_versioned_tasks_have_version_reason():
    """Every task with a .vN suffix must declare version_reason.

    The only exception is a task that uses .v1 as its *initial* name
    (i.e. it has no adapted_from, indicating it was not created to supersede
    a prior version but was simply labelled v1 from the start).
    """
    tasks = mteb.get_tasks(exclude_superseded=False, exclude_beta=False)
    missing = []
    for task in tasks:
        name = task.metadata.name
        if not re.search(r"\.v\d+$", name):
            continue
        # Skip initial-edition tasks: no adapted_from means it was not
        # created as a correction/update of an earlier MTEB task.
        if task.metadata.adapted_from is None:
            continue
        if task.metadata.version_reason is None:
            missing.append(name)

    assert not missing, (
        f"The following versioned tasks are missing version_reason:\n"
        + "\n".join(f"  - {n}" for n in sorted(missing))
    )
