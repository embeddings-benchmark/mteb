from __future__ import annotations

import ast
import sys
from pathlib import Path

from git import Repo


def get_changed_files(base_branch="main"):
    repo_path = Path(__file__).parent.parent
    repo = Repo(repo_path)
    repo.remotes.origin.fetch(base_branch)

    base_commit = repo.commit(f"origin/{base_branch}")
    head_commit = repo.commit("HEAD")

    diff = repo.git.diff("--name-only", base_commit, head_commit)

    changed_files = diff.splitlines()
    return [
        f
        for f in changed_files
        if f.startswith("mteb/models/")
        and f.endswith(".py")
        and "overview" not in f
        and "init" not in f
    ]


def extract_model_names(files: list[str]) -> list[str]:
    model_names = []
    for file in files:
        with open(file) as f:
            tree = ast.parse(f.read())
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if (
                            isinstance(target, ast.Name)
                            and isinstance(node.value, ast.Call)
                            and isinstance(node.value.func, ast.Name)
                            and node.value.func.id == "ModelMeta"
                        ):
                            model_name = next(
                                (
                                    kw.value.value
                                    for kw in node.value.keywords
                                    if kw.arg == "name"
                                ),
                                None,
                            )
                            if model_name:
                                model_names.append(model_name)
    return model_names


if __name__ == "__main__":
    """
    Can pass in base branch as an argument. Defaults to 'main'.
    e.g. python extract_model_names.py mieb
    """
    base_branch = sys.argv[1] if len(sys.argv) > 1 else "main"
    changed_files = get_changed_files(base_branch)
    model_names = extract_model_names(changed_files)
    output_file = Path(__file__).parent / "model_names.txt"
    with output_file.open("w") as f:
        f.write(" ".join(model_names))
