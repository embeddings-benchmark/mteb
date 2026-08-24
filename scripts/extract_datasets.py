import argparse
import ast
import logging
import os
from pathlib import Path

from git import Repo

from scripts.extract_model_names import get_changed_files

logging.basicConfig(level=logging.INFO)


def extract_datasets(files: list[str]) -> list[tuple[str, str]]:
    """Extract dataset (path, revision) tuples from task class files."""
    datasets = []

    for file in files:
        with open(file) as f:
            try:
                tree = ast.parse(f.read())
                for node in ast.walk(tree):
                    # Look for class definitions (task classes)
                    if isinstance(node, ast.ClassDef):
                        # Check if it's a task class by looking for TaskMetadata assignment
                        for class_node in ast.walk(node):
                            if isinstance(class_node, ast.Assign):
                                for target in class_node.targets:
                                    if (
                                        isinstance(target, ast.Name)
                                        and target.id == "metadata"
                                        and isinstance(class_node.value, ast.Call)
                                    ):
                                        # Extract dataset info from TaskMetadata
                                        dataset_info = extract_dataset_from_metadata(
                                            class_node.value
                                        )
                                        if dataset_info:
                                            datasets.append(dataset_info)

                    # Also look for direct dataset dictionary assignments
                    elif isinstance(node, ast.Assign):
                        for target in node.targets:
                            if (
                                isinstance(target, ast.Name)
                                and target.id == "dataset"
                                and isinstance(node.value, ast.Dict)
                            ):
                                dataset_info = extract_dataset_from_dict(node.value)
                                if dataset_info:
                                    datasets.append(dataset_info)

            except SyntaxError as e:
                logging.warning(f"Could not parse {file}: {e}")
                continue

    # Remove duplicates while preserving order
    unique_datasets = list(dict.fromkeys(datasets))

    # Set environment variable in format "path1:revision1,path2:revision2,..."
    if unique_datasets:
        custom_revisions = ",".join(
            f"{path}:{revision}" for path, revision in unique_datasets
        )
        os.environ["CUSTOM_DATASET_REVISIONS"] = custom_revisions
        logging.debug(f"Set CUSTOM_DATASET_REVISIONS={custom_revisions}")

        print(f'export CUSTOM_DATASET_REVISIONS="{custom_revisions}"')
    else:
        # Keep the test module from falling back to the full dataset sweep.
        print('export CUSTOM_DATASET_REVISIONS="__EMPTY__"')
    return unique_datasets


def extract_added_datasets(base_branch: str) -> list[tuple[str, str]]:
    """Extract datasets declared by task files added on the current branch."""
    changed_files = get_changed_files(base_branch, startswith="mteb/tasks/")
    repo = Repo(Path(__file__).parent.parent)
    base_commit = repo.commit(f"origin/{base_branch}")
    head_commit = repo.commit("HEAD")
    added_paths = {
        diff.b_path for diff in base_commit.diff(head_commit, diff_filter="A")
    }
    added_files = [file for file in changed_files if file in added_paths]
    return extract_datasets(added_files)


def extract_dataset_from_metadata(call_node: ast.Call) -> tuple[str, str] | None:
    """Extract dataset info from TaskMetadata call."""
    for keyword in call_node.keywords:
        if (
            keyword.arg == "is_public"
            and isinstance(keyword.value, ast.Constant)
            and not keyword.value.value
        ):
            return None
    for keyword in call_node.keywords:
        if keyword.arg == "dataset" and isinstance(keyword.value, ast.Dict):
            return extract_dataset_from_dict(keyword.value)
    return None


def extract_dataset_from_dict(dict_node: ast.Dict) -> tuple[str, str] | None:
    """Extract path and revision from a dataset dictionary."""
    path = None
    revision = None

    for key, value in zip(dict_node.keys, dict_node.values):
        if isinstance(key, ast.Constant) and key.value == "path":
            if isinstance(value, ast.Constant):
                path = value.value
        elif isinstance(key, ast.Constant) and key.value == "revision":
            if isinstance(value, ast.Constant):
                revision = value.value
        # Handle older Python versions with ast.Str
        elif isinstance(key, ast.Str) and key.s == "path":
            if isinstance(value, ast.Str):
                path = value.s
        elif isinstance(key, ast.Str) and key.s == "revision":
            if isinstance(value, ast.Str):
                revision = value.s

    if path and revision:
        return (path, revision)
    return None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "base_branch",
        nargs="?",
        default="main",
        help="Base branch to compare changes with",
    )
    return parser.parse_args()


if __name__ == "__main__":
    """
    Extract datasets from changed task files compared to a base branch.i

    Can pass in base branch as an argument. Defaults to 'main'.
    e.g. python -m scripts.extract_datasets mieb
    """
    logging.basicConfig(level=logging.INFO)

    args = parse_args()

    base_branch = args.base_branch
    dataset_tuples = extract_added_datasets(base_branch)

    logging.debug(f"Found {len(dataset_tuples)} unique datasets.")
