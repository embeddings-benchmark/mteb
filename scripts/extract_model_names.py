from __future__ import annotations

import argparse
import ast
import logging
from pathlib import Path

from git import Repo

logging.basicConfig(level=logging.INFO)


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
        and "instructions" not in f
        and Path(f).exists()
    ]


def extract_model_names(
    files: list[str], return_one_model_name_per_file=False
) -> list[str]:
    model_names = []
    first_model_found = False
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
                            try:
                                model_name = next(
                                    (
                                        kw.value.value
                                        for kw in node.value.keywords
                                        if kw.arg == "name"
                                    ),
                                    None,
                                )
                            except AttributeError:
                                # For cases where name is assigned a variable and not a direct string,
                                # e.g. in gme_v_models.py: `name=HF_GME_QWEN2VL_2B`
                                model_name = next(
                                    (
                                        kw.value.id
                                        for kw in node.value.keywords
                                        if kw.arg == "name"
                                    ),
                                    None,
                                )
                            if model_name:
                                model_names.append(model_name)
                                first_model_found = True
                if return_one_model_name_per_file and first_model_found:
                    logging.info(f"Found model name {model_name} in file {file}")
                    break  # NOTE: Only take the first model_name per file to avoid disk out of space issue.
    return model_names


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "base_branch",
        nargs="?",
        default="main",
        help="Base branch to compare changes with",
    )
    parser.add_argument(
        "--return_one_model_name_per_file",
        action="store_true",
        default=False,
        help="Only return one model name per file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    """
    Can pass in base branch as an argument. Defaults to 'main'.
    e.g. python extract_model_names.py mieb
    """

    args = parse_args()

    base_branch = args.base_branch
    changed_files = get_changed_files(base_branch)
    model_names = extract_model_names(
        changed_files,
        return_one_model_name_per_file=args.return_one_model_name_per_file,
    )
    output_file = Path(__file__).parent / "model_names.txt"
    with output_file.open("w") as f:
        f.write(" ".join(model_names))
