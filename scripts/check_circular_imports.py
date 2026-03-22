"""Detect circular imports in the mteb package.

Analyzes top-level runtime imports (excluding TYPE_CHECKING blocks and
function-body imports) and reports any circular dependency chains.

Usage:
    python scripts/check_circular_imports.py
"""

from __future__ import annotations

import ast
import os
import sys
from collections import defaultdict
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent.parent / "mteb"

# Self-imports through __init__.py are expected in Python packages
ALLOWED_SELF_IMPORTS = {"mteb"}


def _get_module_name(filepath: Path) -> str | None:
    rel = filepath.relative_to(PACKAGE_DIR.parent)
    s = str(rel)
    if s.endswith("/__init__.py"):
        return s[: -len("/__init__.py")].replace("/", ".")
    if s.endswith(".py"):
        return s[: -len(".py")].replace("/", ".")
    return None


def _resolve_relative_import(
    module: str, level: int, from_module: str | None
) -> str | None:
    parts = module.split(".")
    if level > len(parts):
        return None
    base = ".".join(parts[:-level])
    if from_module:
        return f"{base}.{from_module}" if base else from_module
    return base or None


def _is_type_checking_guard(node: ast.If) -> bool:
    test = node.test
    if isinstance(test, ast.Name) and test.id == "TYPE_CHECKING":
        return True
    if isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING":
        return True
    return False


def build_import_graph() -> dict[str, set[str]]:
    """Build a graph of top-level runtime imports between mteb modules."""
    deps: dict[str, set[str]] = defaultdict(set)

    for root, _dirs, files in os.walk(PACKAGE_DIR):
        for fname in files:
            if not fname.endswith(".py"):
                continue
            filepath = Path(root) / fname
            module = _get_module_name(filepath)
            if not module:
                continue
            try:
                tree = ast.parse(filepath.read_text(), str(filepath))
            except SyntaxError:
                continue

            for node in ast.iter_child_nodes(tree):
                # Skip TYPE_CHECKING blocks
                if isinstance(node, ast.If) and _is_type_checking_guard(node):
                    continue

                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.startswith("mteb"):
                            deps[module].add(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module and node.module.startswith("mteb"):
                        deps[module].add(node.module)
                    elif node.level > 0:
                        resolved = _resolve_relative_import(
                            module, node.level, node.module
                        )
                        if resolved and resolved.startswith("mteb"):
                            deps[module].add(resolved)

    return deps


def find_cycles(graph: dict[str, set[str]]) -> list[list[str]]:
    """Find all cycles in the import graph using DFS."""
    cycles: list[list[str]] = []
    visited: set[str] = set()
    path: list[str] = []
    path_set: set[str] = set()

    def dfs(node: str) -> None:
        visited.add(node)
        path.append(node)
        path_set.add(node)
        for neighbor in sorted(graph.get(node, [])):
            if neighbor in path_set:
                cycle_start = path.index(neighbor)
                cycles.append(path[cycle_start:] + [neighbor])
            elif neighbor not in visited:
                dfs(neighbor)
        path.pop()
        path_set.discard(node)

    for node in sorted(graph):
        if node not in visited:
            dfs(node)

    return cycles


def filter_allowed_cycles(cycles: list[list[str]]) -> list[list[str]]:
    """Remove cycles that are expected/allowed."""
    return [
        cycle
        for cycle in cycles
        if not (len(cycle) == 2 and cycle[0] in ALLOWED_SELF_IMPORTS)
    ]


def main() -> int:
    graph = build_import_graph()
    cycles = find_cycles(graph)
    cycles = filter_allowed_cycles(cycles)

    if not cycles:
        print("No circular imports detected.")
        return 0

    print(f"Found {len(cycles)} circular import chain(s):\n")
    for i, cycle in enumerate(cycles, 1):
        print(f"  {i}. {' -> '.join(cycle)}")
    print(
        "\nTo fix: move the import causing the cycle into the function body where it's used,"
    )
    print("or import from the submodule directly instead of through mteb/__init__.py.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
