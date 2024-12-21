from __future__ import annotations

import ast
import importlib
import inspect
import os
import types
from pathlib import Path

from mteb.abstasks import AbsTask

BASE_DIR = Path("../mteb/tasks")


def find_task_classes_in_module(full_module_name):
    """Import a module and return a list of classes inheriting from AbsTask."""
    try:
        mod = importlib.import_module(full_module_name)
    except ImportError:
        return []

    task_classes = []
    for name, obj in inspect.getmembers(mod, inspect.isclass):
        if (
            isinstance(obj, type)
            and not isinstance(obj, types.GenericAlias)
            and issubclass(obj, AbsTask)
            and obj is not AbsTask
            and not obj.__name__.startswith("AbsTask")
            and not obj.__name__ == "MultilingualTask"
        ):
            task_classes.append(name)
    return task_classes


def parse_all_from_init(init_path):
    """Parse __all__ from an existing __init__.py file to aggregate imports."""
    if not init_path.is_file():
        return []
    with open(init_path) as f:
        tree = ast.parse(f.read())
    all_assignments = [
        n
        for n in tree.body
        if isinstance(n, ast.Assign)
        and len(n.targets) == 1
        and n.targets[0].id == "__all__"
    ]
    if not all_assignments:
        return []
    # Expecting __all__ to be a list of strings
    val = all_assignments[0].value
    if isinstance(val, ast.List):
        return [elt.s for elt in val.elts if isinstance(elt, ast.Str)]
    return []


for root, dirs, files in os.walk(BASE_DIR, topdown=False):
    # Process this directory
    py_files = [f for f in files if f.endswith(".py") and f != "__init__.py"]
    relative_path = Path(root).relative_to(BASE_DIR.parent)
    package_path = ".".join(relative_path.parts)

    # Find classes in Python files of the current directory
    import_lines = []
    all_classes = []
    for py_file in py_files:
        module_name = py_file[:-3]  # remove .py
        full_module_name = f"mteb.{package_path}.{module_name}"
        task_classes = find_task_classes_in_module(full_module_name)
        if task_classes:
            import_line = f"from .{module_name} import {', '.join(task_classes)}"
            import_lines.append(import_line)
            all_classes.extend(task_classes)

    # Also aggregate subdirectories that have their own __init__.py and __all__
    sub_import_lines = []
    for d in dirs:
        sub_init = Path(root) / d / "__init__.py"
        if sub_init.exists():
            sub_all = parse_all_from_init(sub_init)
            if sub_all:
                # Import all from the subpackage
                sub_import_line = f"from .{d} import {', '.join(sub_all)}"
                import_lines.append(sub_import_line)
                all_classes.extend(sub_all)
    # Deduplicate classes
    all_classes = list(
        dict.fromkeys(all_classes)
    )  # preserves order while removing duplicates

    init_path = Path(root) / "__init__.py"
    with open(init_path, "w") as init_file:
        # Write imports from current directory modules
        for line in import_lines:
            init_file.write(line + "\n")

        # Write imports from subdirectories
        for line in sub_import_lines:
            init_file.write(line + "\n")

        # Write __all__
        init_file.write(f"__all__ = {all_classes!r}\n")

    print(f"Updated {init_path} with imports and __all__ = {all_classes}")
