import ast
import importlib.util
import re
import traceback
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import datasets
import orjson
import pandas as pd
import typer
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi
from tqdm.auto import tqdm

"""
This script is designed for data cleaning and the automatic creation of new task versions with updated data.
Currently, it supports only monolingual classification tasks.

The results of this script can be seen in the following PRs:
- https://github.com/embeddings-benchmark/mteb/pull/2632
- https://github.com/embeddings-benchmark/mteb/pull/2900
"""

app = typer.Typer()
datasets.logging.set_verbosity_error()
datasets.logging.disable_progress_bar()


@dataclass
class TaskMetadataInfo:
    class_name: str
    name: str
    version: int
    dataset: dict[str, str]
    eval_splits: list[str]
    eval_langs: list[str]


def format_scores(
    scores_files: list[Path],
    filter_multiple_model_versions: bool = False,
    name_type: Literal["model", "parent"] = "model",
    filter_missing_scores: bool = False,
    use_all_subsets: bool = False,
) -> pd.DataFrame:
    if filter_multiple_model_versions:
        latest_files: dict[tuple[str, str], Path] = {}
        for scores_file in scores_files:
            model_name = (
                scores_file.parent.parent.name
                if name_type == "model"
                else scores_file.parent.parent.parent.name
            )
            task_id = scores_file.name
            key = (model_name, task_id)
            if (
                key not in latest_files
                or scores_file.stat().st_mtime > latest_files[key].stat().st_mtime
            ):
                latest_files[key] = scores_file
        filtered_scores_files = list(latest_files.values())
    else:
        filtered_scores_files = scores_files

    scores_data = []
    for scores_file in filtered_scores_files:
        try:
            s = orjson.loads(scores_file.read_bytes().replace(b"NaN", b"null"))
            if use_all_subsets:
                for subset in s["scores"]:
                    scores_data.extend(
                        {
                            "task_name": s["task_name"],
                            "evaluation_time": s.get(
                                "evaluation_time", None
                            ),  # Handle missing eval time
                            "model_name": scores_file.parent.parent.name
                            if name_type == "model"
                            else scores_file.parent.parent.parent.name,
                            "subset": subset,
                            **score,
                        }
                        for score in s["scores"][subset]
                    )

            else:
                score_set = s["scores"].get(
                    "test", s["scores"].get("dev", s["scores"].get("train"))
                )
                if score_set is None:
                    warnings.warn(
                        f"No 'test' or 'dev' or 'train' scores found in {scores_file}",
                        stacklevel=2,
                    )
                    continue

                scores_data.extend(
                    {
                        "task_name": s["task_name"],
                        "evaluation_time": s.get("evaluation_time", None),
                        "model_name": scores_file.parent.parent.name
                        if name_type == "model"
                        else scores_file.parent.parent.parent.name,
                        **score,
                    }
                    for score in score_set
                )
        except Exception as e:
            warnings.warn(f"Error processing file {scores_file}: {e}", stacklevel=2)

    if not scores_data:
        return pd.DataFrame()

    scores = pd.DataFrame(scores_data)
    scores["languages"] = scores["languages"].apply(
        lambda x: ",".join(x) if isinstance(x, list) else x
    )
    scores["model_name_original"] = scores["model_name"].str.replace("__", "/")

    scores = scores.drop_duplicates(
        subset=["task_name", "model_name", "languages", "hf_subset", "main_score"]
    )
    scores = scores.sort_values(["task_name", "main_score"], ascending=[True, False])

    mode_count = scores.groupby("model_name")["main_score"].count().mode().iloc[0]
    model_counts = scores.groupby("model_name")["main_score"].count()
    filtered_models = model_counts[model_counts < mode_count].index.tolist()

    if filtered_models:
        print(
            f"WARNING: The following models have fewer scores than the mode ({mode_count}):"
        )
        for model in filtered_models:
            print(f"  - {model}: {model_counts[model]} scores")

    if filter_missing_scores:
        scores = scores[~scores["model_name"].isin(filtered_models)]

    return scores


def find_class_node(
    module: ast.Module, class_name: str
) -> tuple[ast.ClassDef | None, int]:
    for i, node in enumerate(module.body):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node, i
    return None, -1


def find_latest_class_name(module: ast.Module) -> str | None:
    version_pattern = re.compile(r"^(?P<base>.+?)(?:V(?P<ver>\d+))?$")
    groups: dict[str, list[tuple[int, ast.ClassDef, int]]] = {}
    for idx, node in enumerate(module.body):
        if isinstance(node, ast.ClassDef):
            m = version_pattern.match(node.name)
            if not m:
                continue
            base = m.group("base")
            ver = int(m.group("ver")) if m.group("ver") else 1
            groups.setdefault(base, []).append((ver, node, idx))
    if not groups:
        return None
    base_class_name, _ = max(groups.items(), key=lambda kv: max(e[0] for e in kv[1]))
    return base_class_name


def read_lines(file_path: Path) -> list[str]:
    return file_path.read_text().splitlines(keepends=True)


def write_lines(file_path: Path, lines: list[str]) -> None:
    file_path.write_text("".join(lines))


def resolve_local_variable(
    var_name: str, func_node: ast.FunctionDef
) -> ast.expr | None:
    # Walk backwards to find the last assignment
    for stmt in reversed(func_node.body):
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if isinstance(target, ast.Name) and target.id == var_name:
                    return stmt.value
    return None


def update_class_header(
    new_block: list[str], old_name: str, base_name: str, new_suffix: str
) -> list[str]:
    header_re = re.compile(r"^(\s*class\s+)" + re.escape(old_name) + r"(\b.*)$")
    new_block[0] = header_re.sub(r"\1" + base_name + new_suffix + r"\2", new_block[0])
    return new_block


def update_task_metadata(
    new_block: list[str], old_name: str, base_name: str, new_meta_suffix: str
) -> list[str]:
    within = False
    parens = 0
    metadata_start_index = -1
    metadata_end_index = -1
    indent = -1
    adapted_from_index = -1

    for i, line in enumerate(new_block):
        if not within and re.match(r"\s*metadata\s*=\s*TaskMetadata\s*\(", line):
            within = True
            metadata_start_index = i
            parens = line.count("(") - line.count(")")
            if line.strip().endswith("()"):
                parens = 0
                metadata_end_index = i
                within = False
        elif within:
            if "adapted_from" in line:
                adapted_from_index = i
            parens += line.count("(") - line.count(")")
            m = re.match(r'^(\s*name\s*=\s*")([^"]*)(".*)$', line)
            if m:
                base_name = re.sub(r"\.v\d+$", "", m.group(2))
                new_block[i] = (
                    m.group(1) + base_name + new_meta_suffix + m.group(3) + "\n"
                )
            if indent == -1 and line.strip() and not line.strip().startswith("#"):
                indent = len(line) - len(line.lstrip(" "))
            if parens <= 0:
                within = False
                metadata_end_index = i

    if metadata_start_index != -1:
        if adapted_from_index != -1:
            line = new_block[adapted_from_index]
            line_indent = len(line) - len(line.lstrip(" "))
            new_block[adapted_from_index] = (
                f'{" " * line_indent}adapted_from=["{old_name}"],\n'
            )
        else:
            if indent == -1:
                line = new_block[metadata_end_index]
                indent = (len(line) - len(line.lstrip())) + 4

            line_to_insert_before_idx = metadata_end_index
            # handle case where last line is just ')'
            if new_block[line_to_insert_before_idx].strip() == ")":
                line_to_insert_before_idx -= 1

            last_line_idx = -1
            for i in range(line_to_insert_before_idx, -1, -1):
                if new_block[i].strip():
                    last_line_idx = i
                    break

            if last_line_idx != -1:
                if not new_block[last_line_idx].rstrip().endswith(","):
                    new_block[last_line_idx] = new_block[last_line_idx].rstrip() + ",\n"

            new_line = " " * indent + f'adapted_from=["{old_name}"],\n'
            new_block.insert(metadata_end_index, new_line)
    return new_block


def handle_dataset_transform(
    new_block: list[str], block: list[str], ds: DatasetDict
) -> list[str]:
    text = "".join(new_block)
    module = ast.parse(text)
    class_node = next(node for node in module.body if isinstance(node, ast.ClassDef))

    transform_node = None
    for node in class_node.body:
        if isinstance(node, ast.FunctionDef) and node.name == "dataset_transform":
            transform_node = node
            break

    if transform_node:
        subsampling_calls = []
        for node in ast.walk(transform_node):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "stratified_subsampling"
            ):
                subsampling_calls.append(node)

        if not subsampling_calls:
            start = transform_node.lineno - 1
            end = transform_node.end_lineno
            del new_block[start:end]
        else:
            n_samples = 2048
            splits = ["test"]
            for call in subsampling_calls:
                for kw in call.keywords:
                    if kw.arg == "n_samples":
                        value_node = kw.value
                        if isinstance(value_node, ast.Name):
                            resolved_node = resolve_local_variable(
                                value_node.id, transform_node
                            )
                            if resolved_node:
                                value_node = resolved_node
                        try:
                            n_samples = ast.literal_eval(value_node)
                        except (ValueError, TypeError):
                            pass  # Keep default
                    elif kw.arg == "splits":
                        splits = ast.literal_eval(kw.value)

            needs_subsampling = any(
                len(ds[split]) > n_samples for split in splits if split in ds
            )

            if needs_subsampling:
                original_class_lines = "".join(block)
                original_class_module = ast.parse(original_class_lines)
                original_class_node = next(
                    node
                    for node in original_class_module.body
                    if isinstance(node, ast.ClassDef)
                )

                original_transform_node = None
                for node in original_class_node.body:
                    if (
                        isinstance(node, ast.FunctionDef)
                        and node.name == "dataset_transform"
                    ):
                        original_transform_node = node
                        break

                new_transform_body = []
                if original_transform_node:
                    for stmt in original_transform_node.body:
                        is_subsampling_stmt = False
                        for node in ast.walk(stmt):
                            if (
                                isinstance(node, ast.Call)
                                and isinstance(node.func, ast.Attribute)
                                and node.func.attr == "stratified_subsampling"
                            ):
                                is_subsampling_stmt = True
                                break
                        if is_subsampling_stmt:
                            stmt_start_line = stmt.lineno - 1
                            stmt_end_line = (
                                stmt.end_lineno
                                if stmt.end_lineno
                                else stmt_start_line + 1
                            )
                            new_transform_body.extend(
                                block[stmt_start_line:stmt_end_line]
                            )

                start = (
                    transform_node.body[0].lineno - 1
                    if transform_node.body
                    else transform_node.lineno
                )
                end = transform_node.end_lineno
                del new_block[start:end]

                if new_transform_body:
                    indent = (
                        " " * (transform_node.body[0].col_offset)
                        if transform_node.body
                        else " " * (transform_node.col_offset + 4)
                    )
                    new_transform_body_lines = [
                        f"{indent}{line.lstrip()}" for line in new_transform_body
                    ]
                    new_block.insert(start, "".join(new_transform_body_lines))

            else:
                start = transform_node.lineno - 1
                end = transform_node.end_lineno
                del new_block[start:end]
    return new_block


def get_v2_block(
    block: list[str],
    old_name: str,
    base_name: str,
    new_suffix: str,
    new_meta_suffix: str,
    ds: DatasetDict,
) -> list[str]:
    new_block = block.copy()
    new_block = update_class_header(new_block, old_name, base_name, new_suffix)
    new_block = update_task_metadata(new_block, old_name, base_name, new_meta_suffix)
    new_block = handle_dataset_transform(new_block, block, ds)
    return new_block


def deduplicate(dataset: Dataset) -> Dataset:
    unique_texts = set()
    indices_to_keep = []
    for i, text in enumerate(dataset["text"]):
        text = text.strip()
        if text not in unique_texts:
            unique_texts.add(text)
            indices_to_keep.append(i)

    return dataset.select(indices_to_keep)


def filter_empty(dataset: Dataset) -> Dataset:
    return dataset.filter(lambda x: len(x["text"].strip()) > 0)


def filter_leakage(train_dataset: Dataset, test_dataset: Dataset) -> Dataset:
    train_texts = set(train_dataset["text"])
    test_indices_no_leakage = [
        i for i, text in enumerate(test_dataset["text"]) if text not in train_texts
    ]
    return test_dataset.select(test_indices_no_leakage)


def filter_controversial(dataset_dict: DatasetDict) -> DatasetDict:
    normalized: dict[str, set[str | tuple[str, ...]]] = {}
    for _, ds in dataset_dict.items():
        for text, label in zip(ds["text"], ds["label"]):
            key = text.strip().lower()
            normalized.setdefault(key, set()).add(
                label if isinstance(label, str | int | float) else tuple(label)
            )
    bad_texts = {t for t, labels in normalized.items() if len(labels) > 1}
    return DatasetDict(
        {
            split: ds.filter(lambda x: x["text"].strip().lower() not in bad_texts)
            for split, ds in dataset_dict.items()
        }
    )


def filter_short(dataset: Dataset, min_words: int = 3) -> Dataset:
    return dataset.filter(lambda x: len(x["text"].strip().split()) >= min_words)


def calculate_inner_indent(lines: list[str], node: ast.ClassDef) -> int:
    indent = len(lines[node.lineno - 1]) - len(lines[node.lineno - 1].lstrip(" "))
    for line in lines[node.lineno : node.end_lineno]:
        if line.strip():
            return len(line) - len(line.lstrip(" "))
    return indent + 4


def load_and_transform(file_path: Path, metadata: TaskMetadataInfo) -> DatasetDict:
    return load_dataset(file_path, metadata.class_name)


def split_train_test(
    ds: DatasetDict, metadata: TaskMetadataInfo
) -> tuple[DatasetDict, bool, list[tuple[str, str, int]]]:
    report: list[tuple[str, str, int]] = []
    is_changed = False
    if "train" in ds and metadata.eval_splits == "train":
        is_changed = True
        before = len(ds["train"])
        ds["train"] = ds["train"].cast_column(
            "label", datasets.ClassLabel(names=list(set(ds["train"]["label"])))
        )
        label_counts = pd.Series(ds["train"]["label"]).value_counts()
        one_sample_labels = set(label_counts[label_counts == 1].index.tolist())

        if len(one_sample_labels) > 0:
            before_size = len(ds["train"])
            ds["train"] = ds["train"].filter(
                lambda x: x["label"] not in one_sample_labels
            )
            removed = before_size - len(ds["train"])
            if removed > 0:
                report.append(("filter_one_sample_labels", "train", removed))

        splits = ds["train"].train_test_split(
            test_size=min(2048, before // 2), seed=42, stratify_by_column="label"
        )
        ds = DatasetDict({"train": splits["train"], "test": splits["test"]})
        report.append(("create_test_split", "train_to_test", before - len(ds["train"])))
        metadata.eval_splits = ["test"]
    return ds, is_changed, report


def clean_dataset(
    ds: DatasetDict,
    metadata: TaskMetadataInfo,
) -> tuple[DatasetDict, list[tuple[str, str, int]], bool]:
    report: list[tuple[str, str, int]] = []
    is_changed = False

    skip_codes = {"zho", "jpn", "tha", "mya", "cmn"}
    apply_short = not any(
        lang.split("-")[0] in skip_codes for lang in metadata.eval_langs
    )

    transforms = [
        ("filter_empty", filter_empty),
        ("deduplicate", deduplicate),
    ]
    if apply_short:
        transforms.append(("filter_short", filter_short))

    for split in ["train", *metadata.eval_splits]:
        if split not in ds:
            continue
        for name, fn in transforms:
            before = len(ds[split])
            ds[split] = fn(ds[split])
            removed = before - len(ds[split])
            if removed > 0:
                is_changed = True
                report.append((name, split, removed))

    ds, is_changed_after_split, split_report = split_train_test(ds, metadata)
    report.extend(split_report)
    is_changed = is_changed or is_changed_after_split

    for split in metadata.eval_splits:
        if split == "train":
            continue
        before_test = len(ds[split])
        ds["test"] = filter_leakage(ds["train"], ds[split])
        removed = before_test - len(ds[split])
        if removed > 0:
            is_changed = True
            report.append(("filter_leakage", split, removed))

    orig = {split: len(ds[split]) for split in ds}
    ds = filter_controversial(ds)
    for split in ds:
        removed = orig[split] - len(ds[split])
        if removed > 0:
            is_changed = True
            report.append(("filter_controversial", split, removed))

    return ds, report, is_changed


def print_report(
    report_folder: Path,
    language: str,
    original_records: list[tuple[str, str, int]],
    filter_records: list[tuple[str, str, str, int]],
) -> None:
    report_lines: list[str] = []
    report_lines.append("## Original Sizes")
    report_lines.append("| Task | Split | Original Size |")
    report_lines.append("|------|:-----:|--------------:|")
    for task, split, size in original_records:
        report_lines.append(f"| {task} | {split} | {size} |")

    report_lines.append("")
    report_lines.append("## Cleaning Report")
    report_lines.append("| Task | Filter | Split | Removed |")
    report_lines.append("|------|--------|:-----:|--------:|")
    for task, name, split, removed in filter_records:
        report_lines.append(f"| {task} | {name} | {split} | {removed} |")

    (report_folder / f"report_{language}.md").write_text("\n".join(report_lines))


def push_dataset(ds: DatasetDict, metadata: TaskMetadataInfo, username: str) -> str:
    prev_path = metadata.dataset.get("path", "")
    if prev_path.startswith("mteb/") and username != "mteb":
        repo_id = prev_path.replace("mteb/", f"{username}/")
    else:
        base = metadata.class_name
        if base.endswith("Classification"):
            base = base[: -len("Classification")]
        name = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", base)
        name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name).lower()
        repo_id = f"{username}/{name}"
    ds.push_to_hub(repo_id, config_name=metadata.dataset.get("name", "default"))
    return repo_id


def update_metadata(
    file_path: Path, class_name: str, new_ver: int, repo_id: str, pr_id: int
) -> None:
    api = HfApi()
    commit = api.list_repo_commits(repo_id=repo_id, repo_type="dataset")[0].commit_id
    update_v2_metadata_dataset(
        file_path, class_name + f"V{new_ver}", repo_id, commit, pr_id
    )


def parse_metadata_dataset(file_path: Path, class_name: str) -> dict[str, str]:
    source = file_path.read_text()
    module = ast.parse(source)
    class_node, _ = find_class_node(module, class_name)
    for node in class_node.body:
        if isinstance(node, ast.Assign):
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id == "metadata":
                call = node.value
                for kw in call.keywords:
                    if kw.arg == "dataset":
                        return ast.literal_eval(kw.value)
    return {}


def get_transform_statements(file_path: Path, class_name: str) -> list[ast.stmt]:
    source = file_path.read_text()
    module = ast.parse(source)
    class_node, _ = find_class_node(module, class_name)
    for node in class_node.body:
        if isinstance(node, ast.FunctionDef) and node.name == "dataset_transform":
            return [
                stmt
                for stmt in node.body
                if not (
                    isinstance(stmt, ast.Assign)
                    and isinstance(stmt.value, ast.Call)
                    and isinstance(stmt.value.func, ast.Attribute)
                    and stmt.value.func.attr == "stratified_subsampling"
                )
            ]
    return []


def load_dataset(file_path: Path, class_name: str) -> DatasetDict:
    original = file_path.read_text()
    lines = original.splitlines(keepends=True)
    filtered: list[str] = []
    skip = False
    parens = 0

    for line in lines:
        if not skip and "stratified_subsampling" in line:
            skip = True
            parens = line.count("(") - line.count(")")
            continue
        if skip:
            parens += line.count("(") - line.count(")")
            if parens <= 0:
                skip = False
                filtered.append("        pass")
            continue
        filtered.append(line)

    file_path.write_text("".join(filtered))

    spec = importlib.util.spec_from_file_location("task_module", str(file_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    TaskClass = getattr(module, class_name)
    task = TaskClass()
    task.load_data()
    ds = task.dataset

    file_path.write_text(original)

    return ds


def _find_metadata_assignment(class_node: ast.ClassDef) -> ast.Assign | None:
    for stmt in class_node.body:
        if (
            isinstance(stmt, ast.Assign)
            and isinstance(stmt.targets[0], ast.Name)
            and stmt.targets[0].id == "metadata"
        ):
            return stmt
    return None


def _find_keyword(call_node: ast.Call, keyword_name: str) -> ast.keyword | None:
    for kw in call_node.keywords:
        if kw.arg == keyword_name:
            return kw
    return None


def _get_indent(line: str) -> str:
    return line[: len(line) - len(line.lstrip())]


def _update_description(
    lines: list[str], call_node: ast.Call, pr_id: int
) -> tuple[list[str], list[int]]:
    desc_kw = _find_keyword(call_node, "description")
    if not desc_kw or not isinstance(desc_kw.value, ast.Constant):
        return lines, []

    value_node = desc_kw.value
    original_desc = value_node.value
    start_line_idx = desc_kw.lineno - 1
    end_line_idx = value_node.end_lineno - 1

    indent = _get_indent(lines[start_line_idx])
    new_desc_val = f'"""{original_desc}\n        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/{pr_id})"""'
    lines[start_line_idx] = f"{indent}description={new_desc_val},\n"

    deleted_indices = list(range(start_line_idx + 1, end_line_idx + 1))
    return lines, deleted_indices


def _update_dataset_dict(
    lines: list[str], call_node: ast.Call, new_path: str, new_revision: str
) -> tuple[list[str], list[int]]:
    dataset_kw = _find_keyword(call_node, "dataset")
    if not dataset_kw or not isinstance(dataset_kw.value, ast.Dict):
        return lines, []

    dict_node = dataset_kw.value
    lines_to_delete = []

    for i, key_node in enumerate(dict_node.keys):
        if not isinstance(key_node, ast.Constant):
            continue

        value_node = dict_node.values[i]
        line_idx = key_node.lineno - 1
        indent = _get_indent(lines[line_idx])
        key = key_node.value

        if key == "path":
            lines[line_idx] = f'{indent}"path": "{new_path}",\n'
        elif key == "revision":
            lines[line_idx] = f'{indent}"revision": "{new_revision}",\n'
        elif key == "trust_remote_code":
            lines_to_delete.extend(range(line_idx, value_node.end_lineno))

    return lines, lines_to_delete


def _update_eval_splits(
    lines: list[str], call_node: ast.Call, module: ast.Module
) -> list[str]:
    eval_splits_kw = _find_keyword(call_node, "eval_splits")
    if not eval_splits_kw:
        return lines

    value_node = eval_splits_kw.value
    if isinstance(value_node, ast.Name):
        resolved = _resolve_variable(value_node.id, module)
        if resolved:
            value_node = resolved

    is_train_split = (
        isinstance(value_node, ast.List)
        and len(value_node.elts) == 1
        and isinstance(value_node.elts[0], ast.Constant)
        and value_node.elts[0].value == "train"
    )

    if is_train_split:
        line_idx = eval_splits_kw.lineno - 1
        indent = _get_indent(lines[line_idx])
        lines[line_idx] = f'{indent}eval_splits=["test"],\n'

    return lines


def update_v2_metadata_dataset(
    file_path: Path, class_name: str, new_path: str, new_revision: str, pr_id: int
) -> None:
    lines = read_lines(file_path)
    module = ast.parse("".join(lines))

    class_node, _ = find_class_node(module, class_name)
    if not class_node:
        raise ValueError(f"Class {class_name} not found in {file_path}")

    metadata_node = _find_metadata_assignment(class_node)
    if not metadata_node or not isinstance(metadata_node.value, ast.Call):
        return

    call_node = metadata_node.value
    lines, desc_deleted = _update_description(lines, call_node, pr_id)
    lines, ds_deleted = _update_dataset_dict(lines, call_node, new_path, new_revision)
    lines = _update_eval_splits(lines, call_node, module)

    all_deleted_indices = sorted(set(desc_deleted + ds_deleted), reverse=True)
    for i in all_deleted_indices:
        del lines[i]

    write_lines(file_path, lines)


def _resolve_variable(name: str, module: ast.Module) -> ast.expr | None:
    for node in module.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return node.value
    return None


def parse_all_task_metadata(
    file_path: Path, latest_version: bool = True
) -> list[TaskMetadataInfo]:
    source = file_path.read_text()
    module = ast.parse(source)

    version_pattern = re.compile(r"^(?P<base>.+?)(?:V(?P<ver>\d+))?$")
    all_tasks: list[TaskMetadataInfo] = []

    for node in module.body:
        if not isinstance(node, ast.ClassDef):
            continue

        m = version_pattern.match(node.name)
        if not m:
            continue
        ver = int(m.group("ver")) if m.group("ver") else 1

        name = ""
        dataset: dict[str, str] = {}
        eval_split: list[str] = ["test"]
        eval_langs: list[str] = []
        for stmt in node.body:
            if not (
                isinstance(stmt, ast.Assign)
                and isinstance(stmt.targets[0], ast.Name)
                and stmt.targets[0].id == "metadata"
            ):
                continue
            if not isinstance(stmt.value, ast.Call):
                continue
            call = stmt.value
            for kw in call.keywords:
                value_node = kw.value
                if isinstance(value_node, ast.Name):
                    resolved_node = _resolve_variable(value_node.id, module)
                    if resolved_node:
                        value_node = resolved_node
                try:
                    if kw.arg == "name" and isinstance(value_node, ast.Constant):
                        name = value_node.value
                    elif kw.arg == "dataset":
                        dataset = ast.literal_eval(value_node)
                    elif kw.arg == "eval_splits":
                        eval_split = ast.literal_eval(value_node) or ["test"]
                    elif kw.arg == "eval_langs":
                        eval_langs = ast.literal_eval(value_node)
                except (ValueError, SyntaxError):
                    pass
            break

        if not name:
            continue

        all_tasks.append(
            TaskMetadataInfo(node.name, name, ver, dataset, eval_split, eval_langs)
        )

    if not latest_version:
        return all_tasks

    latest: dict[str, TaskMetadataInfo] = {}
    for task in all_tasks:
        base_name = re.sub(r"V\d+$", "", task.name)
        if base_name not in latest or task.version > latest[base_name].version:
            latest[base_name] = task

    return list(latest.values())


def parse_all_task_metadata_versions(file_path: Path) -> list[TaskMetadataInfo]:
    return parse_all_task_metadata(file_path, latest_version=False)


def bump_version_for_class(
    file_path: Path, base_class_name: str, ds: DatasetDict
) -> int:
    lines = read_lines(file_path)
    module = ast.parse("".join(lines))

    version_pattern = re.compile(rf"^{re.escape(base_class_name)}(?:V(?P<ver>\d+))?$")
    selected: tuple[int, ast.ClassDef] | None = None
    for node in module.body:
        if not isinstance(node, ast.ClassDef):
            continue
        m = version_pattern.match(node.name)
        if not m:
            continue
        ver = int(m.group("ver")) if m.group("ver") else 1
        if selected is None or ver > selected[0]:
            selected = (ver, node)
    if selected is None:
        raise ValueError(f"Class {base_class_name} not found in {file_path}")

    version, node = selected
    inner = calculate_inner_indent(lines, node)
    new_version = version + 1

    task_name = ""
    for stmt in node.body:
        if (
            isinstance(stmt, ast.Assign)
            and isinstance(stmt.targets[0], ast.Name)
            and stmt.targets[0].id == "metadata"
        ):
            call = stmt.value
            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id == "TaskMetadata"
            ):
                for kw in call.keywords:
                    if kw.arg == "name" and isinstance(kw.value, ast.Constant):
                        task_name = kw.value.value
                        break
            break

    superseded = " " * inner + f'superseded_by = "{task_name}.v{new_version}"\n'
    block = lines[node.lineno - 1 : node.end_lineno]
    v2_block = get_v2_block(
        block, node.name, base_class_name, f"V{new_version}", f".v{new_version}", ds
    )

    new_lines: list[str] = []
    for i, l in enumerate(lines):
        new_lines.append(l)
        if i == node.lineno - 1:
            new_lines.append(superseded)
        if i == node.end_lineno - 1:
            new_lines.append("\n")
            new_lines.extend(v2_block)
    write_lines(file_path, new_lines)
    return new_version


def process_task(
    file_path: Path,
    metadata: TaskMetadataInfo,
    pr_id: int,
    username: str,
    verbose: bool,
) -> tuple[
    tuple[str, int] | None,
    list[tuple[str, str, int]],
    list[tuple[str, str, str, int]],
]:
    if verbose:
        print("  task ->", metadata.class_name)
    try:
        ds = load_and_transform(file_path, metadata)
    except Exception:
        print(metadata.class_name, "dataset loading failed")
        traceback.print_exc()
        return None, [], []

    if verbose:
        print(ds)

    original_size = {split: len(ds[split]) for split in ds}
    ds_cleaned, report, is_changed = clean_dataset(ds.copy(), metadata)
    if verbose:
        print(f"is_changed: {is_changed}")

    if not is_changed:
        if verbose:
            print(f"{metadata.class_name} is unchanged")
        return None, [], []

    original_records = [
        (metadata.name, split, size) for split, size in original_size.items()
    ]
    filter_records = [
        (metadata.name, name, split, removed) for name, split, removed in report
    ]

    repo_id = push_dataset(ds_cleaned, metadata, username)
    base_name = re.sub(r"V\d+$", "", metadata.class_name)
    new_ver = bump_version_for_class(file_path, base_name, ds)
    update_metadata(file_path, base_name, new_ver, repo_id, pr_id)

    return (metadata.name, new_ver), original_records, filter_records


@app.command()
def create_and_prepare(
    folder: Path = typer.Argument(..., exists=True, dir_okay=True),
    pr_id: int = typer.Argument(..., help="Pull request ID"),
    report_folder: Path = typer.Option(
        "scripts/data/cleaning_reports", exists=True, dir_okay=True
    ),
    username: str = "mteb",
    start_lang: str | None = None,
    verbose: bool = typer.Option(False, "--verbose"),
) -> None:
    changed_tasks: list[tuple[str, int]] = []
    all_original_records: list[tuple[str, str, int]] = []
    all_filter_records: list[tuple[str, str, str, int]] = []

    files_to_process = sorted(
        p for p in folder.glob("**/*.py") if p.name != "__init__.py"
    )
    if start_lang:
        files_to_process = [p for p in files_to_process if p.parent.name >= start_lang]
    progress_bar = tqdm(files_to_process, desc="Processing files")

    try:
        for file_path in progress_bar:
            progress_bar.set_description(f"Processing {file_path.name}")
            if verbose:
                print("working on", file_path.name)

            for metadata in parse_all_task_metadata(file_path):
                changed_task, original_records, filter_records = process_task(
                    file_path, metadata, pr_id, username, verbose
                )
                if changed_task:
                    changed_tasks.append(changed_task)
                    all_original_records.extend(original_records)
                    all_filter_records.extend(filter_records)
    except Exception:
        print(traceback.format_exc())

    if changed_tasks:
        print_report(
            report_folder, folder.name, all_original_records, all_filter_records
        )

        unique_changed = sorted(set(changed_tasks))
        tasks_str = " ".join(
            f"{task_name} {task_name}.v{version}"
            for task_name, version in unique_changed
        )
        print(
            "mteb run -m intfloat/multilingual-e5-small -t"
            f" {tasks_str} && mteb run -m"
            " sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 -t"
            f" {tasks_str}"
        )


@app.command()
def compare_results(
    results_dir: Path = typer.Option(
        "/home/admin/vatolin/experiments/mteb/results", exists=True, dir_okay=True
    ),
    tasks_file: Path | None = typer.Option(
        None,
        "--tasks-file",
        "-f",
        exists=True,
        file_okay=True,
        dir_okay=False,
        help="File with a list of tasks to compare. One task per line.",
    ),
) -> None:
    models = [
        "intfloat/multilingual-e5-small",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    ]
    scores_files = [
        f for f in results_dir.glob("**/**/*.json") if f.stem not in {"model_meta"}
    ]
    df_all: pd.DataFrame = format_scores(scores_files, use_all_subsets=True)

    required_tasks: set[str] | None = None
    if tasks_file:
        required_tasks = {
            line.strip() for line in tasks_file.read_text().splitlines() if line.strip()
        }

    for model in models:
        df_model = df_all[df_all["model_name"] == model.replace("/", "__")]

        df_old = df_model[
            ~df_model["task_name"].str.contains(r"\.v\d+$", regex=True)
        ].copy()
        df_new = df_model[
            df_model["task_name"].str.contains(r"\.v\d+$", regex=True)
        ].copy()

        df_new["task_name"] = df_new["task_name"].str.replace(
            r"\.v\d+$", "", regex=True
        )

        if required_tasks:
            available_for_comparison = set(df_old["task_name"]).intersection(
                set(df_new["task_name"])
            )
            missing_tasks = required_tasks - available_for_comparison
            if missing_tasks:
                print(f"**{model}**")
                print(
                    f"Skipping due to missing tasks: {', '.join(sorted(missing_tasks))}"
                )
                print()
                continue

            df_old = df_old[df_old["task_name"].isin(required_tasks)]
            df_new = df_new[df_new["task_name"].isin(required_tasks)]

        old_duplicated = df_old.duplicated(subset=["task_name", "subset"])
        if old_duplicated.sum() > 0:
            print("Duplicated scores")
            print(model)
            print(df_old[old_duplicated][["task_name", "subset", "languages"]])
            continue
        new_duplicated = df_new.duplicated(subset=["task_name", "subset"])
        if new_duplicated.sum() > 0:
            print(model)
            print(df_new[new_duplicated][["task_name", "subset", "languages"]])
            continue
        df_old = df_old.set_index(["task_name", "subset"])["main_score"]
        df_new = df_new.set_index(["task_name", "subset"])["main_score"]
        df_cmp = pd.DataFrame(
            {
                "main_score_old": df_old,
                "main_score_new": df_new,
            }
        ).dropna()

        if df_cmp.empty:
            continue

        df_cmp["delta_percent"] = (
            (df_cmp["main_score_new"] - df_cmp["main_score_old"])
            / df_cmp["main_score_old"]
            * 100
        ).round(2)
        df_cmp = df_cmp.reset_index(drop=False).sort_values(["task_name", "subset"])

        print(f"**{model}**")
        print(
            df_cmp[
                [
                    "task_name",
                    "subset",
                    "main_score_old",
                    "main_score_new",
                    "delta_percent",
                ]
            ].to_markdown(index=False)
        )
        print()


@app.command()
def report_cleaning(
    tasks: list[str] = typer.Argument(
        ..., help="List of task names to generate cleaning reports for."
    ),
    folder: Path = typer.Option(
        "mteb/tasks/Classification", exists=True, dir_okay=True
    ),
    report_folder: Path = typer.Option(
        "scripts/data/cleaning_reports", exists=True, dir_okay=True
    ),
    verbose: bool = typer.Option(False, "--verbose"),
) -> None:
    all_original_records: list[tuple[str, str, int]] = []
    all_filter_records: list[tuple[str, str, str, int]] = []

    all_tasks_map: dict[str, tuple[TaskMetadataInfo, Path]] = {}
    files_to_process = sorted(
        p for p in folder.glob("**/*.py") if p.name != "__init__.py"
    )

    for file_path in files_to_process:
        tasks_in_file = parse_all_task_metadata_versions(file_path)
        for task_metadata in tasks_in_file:
            all_tasks_map[task_metadata.name] = (task_metadata, file_path)

    for task_name in tasks:
        v2_task_name = f"{task_name}.v2"
        if v2_task_name not in all_tasks_map:
            if verbose:
                print(f"Task {v2_task_name} not found, skipping.")
            continue

        if task_name not in all_tasks_map:
            if verbose:
                print(f"Base task {task_name} not found, skipping.")
            continue

        v1_metadata, file_path = all_tasks_map[task_name]
        if verbose:
            print(f"Processing {task_name} from {file_path}")

        try:
            ds = load_and_transform(file_path, v1_metadata)
            print(ds)
        except Exception:
            print(f"Dataset loading failed for {v1_metadata.class_name}")
            traceback.print_exc()
            continue

        original_size = {split: len(ds[split]) for split in ds}
        ds_new, report, _ = clean_dataset(ds.copy(), v1_metadata)
        print(ds_new)
        print(report)

        original_records = [
            (v1_metadata.name, split, size) for split, size in original_size.items()
        ]
        filter_records = [
            (v1_metadata.name, name, split, removed) for name, split, removed in report
        ]

        all_original_records.extend(original_records)
        all_filter_records.extend(filter_records)

    if all_filter_records:
        print_report(
            report_folder, folder.name, all_original_records, all_filter_records
        )
        print(f"Report generated in {report_folder}/report_{folder.name}.md")
    else:
        print("No tasks with v2 versions found or no changes after cleaning.")


if __name__ == "__main__":
    app()
