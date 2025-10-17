"""tests for the MTEB CLI"""

import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pytest
import yaml

from mteb.cli.build_cli import (
    _available_benchmarks,
    _available_tasks,
    _create_meta,
    run,
)


def test_available_tasks(capsys):
    args = Namespace(
        categories=None,
        task_types=None,
        languages=None,
        tasks=None,
    )
    _available_tasks(args=args)

    captured = capsys.readouterr()
    assert "LccSentimentClassification" in captured.out, (
        "Sample task LccSentimentClassification task not found in available tasks"
    )


def test_available_benchmarks(capsys):
    args = Namespace(benchmarks=None)
    _available_benchmarks(args=args)

    captured = capsys.readouterr()
    assert "MTEB(eng, v1)" in captured.out, (
        "Sample benchmark MTEB(eng, v1) task not found in available benchmarks"
    )


run_task_fixures = [
    (
        "sentence-transformers/all-MiniLM-L6-v2",
        "BornholmBitextMining",
        "8b3219a92973c328a8e22fadcfa821b5dc75636a",
    ),
]


@pytest.mark.parametrize("model_name,task_name,model_revision", run_task_fixures)
def test_run_task(
    model_name: str,
    task_name: str,
    model_revision: str,
    tmp_path: Path,
):
    args = Namespace(
        model=model_name,
        tasks=[task_name],
        model_revision=model_revision,
        output_folder=tmp_path.as_posix(),
        device=None,
        categories=None,
        task_types=None,
        languages=None,
        batch_size=None,
        verbosity=3,
        co2_tracker=None,
        overwrite_strategy="always",
        eval_splits=None,
        prediction_folder=None,
        benchmarks=None,
        overwrite=False,
        save_predictions=None,
    )

    run(args)

    model_name_as_path = model_name.replace("/", "__").replace(" ", "_")
    results_path = tmp_path / "results" / model_name_as_path / model_revision
    assert results_path.exists(), "Output folder not created"
    assert "model_meta.json" in [f.name for f in list(results_path.glob("*.json"))], (
        "model_meta.json not found in output folder"
    )
    assert f"{task_name}.json" in [f.name for f in list(results_path.glob("*.json"))], (
        f"{task_name} not found in output folder"
    )


def test_create_meta(tmp_path):
    """Test create_meta function directly as well as through the command line interface"""
    test_folder = Path(__file__).parent
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    output_folder = test_folder / "create_meta"
    output_path = tmp_path / "model_card.md"

    args = Namespace(
        model_name=model_name,
        results_folder=output_folder,
        output_path=output_path,
        overwrite=True,
        from_existing=None,
        tasks=None,
        benchmarks=None,
    )

    _create_meta(args)

    assert output_path.exists(), "Output file not created"

    with output_path.open("r") as f:
        meta = f.read()
        meta = meta[meta.index("---") + 3 : meta.index("---", meta.index("---") + 3)]
        frontmatter = yaml.safe_load(meta)

    with (output_folder / "model_card_gold.md").open("r") as f:
        gold = f.read()
        gold = gold[gold.index("---") + 3 : gold.index("---", gold.index("---") + 3)]
        frontmatter_gold = yaml.safe_load(gold)

    # compare the frontmatter (ignoring the order of keys and other elements)
    for key in frontmatter_gold:
        assert key in frontmatter, f"Key {key} not found in output"

        assert frontmatter[key] == frontmatter_gold[key], (
            f"Value for {key} does not match"
        )

    # ensure that the command line interface works as well
    command = f"{sys.executable} -m mteb create-model-results --model-name {model_name} --results-folder {output_folder.as_posix()} --output-path {output_path.as_posix()} --overwrite"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    assert result.returncode == 0, "Command failed"


@pytest.mark.parametrize(
    "existing_readme_name, gold_readme_name",
    [
        ("existing_readme.md", "model_card_gold_existing.md"),
        ("model_card_without_frontmatter.md", "model_card_gold_without_frontmatter.md"),
    ],
)
def test_create_meta_from_existing(
    existing_readme_name: str, gold_readme_name: str, tmp_path: Path
):
    """Test create_meta function directly as well as through the command line interface"""
    test_folder = Path(__file__).parent
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    output_folder = test_folder / "create_meta"
    output_path = tmp_path / "model_card.md"
    existing_readme = output_folder / existing_readme_name

    args = Namespace(
        model_name=model_name,
        results_folder=output_folder,
        output_path=output_path,
        overwrite=True,
        from_existing=str(existing_readme),
        tasks=None,
        benchmarks=None,
    )

    _create_meta(args)

    assert output_path.exists(), "Output file not created"

    yaml_start_sep = "---"
    yaml_end_sep = "\n---\n"  # newline to avoid matching "---" in the content

    with output_path.open("r") as f:
        meta = f.read()
        start_yaml = meta.index(yaml_start_sep) + len(yaml_start_sep)
        end_yaml = meta.index(yaml_end_sep, start_yaml)
        readme_output = meta[end_yaml + len(yaml_end_sep) :]
        meta = meta[start_yaml:end_yaml]
        frontmatter = yaml.safe_load(meta)

    with (output_folder / gold_readme_name).open("r") as f:
        gold = f.read()
        start_yaml = gold.index(yaml_start_sep) + len(yaml_start_sep)
        end_yaml = gold.index(yaml_end_sep, start_yaml)
        gold_readme = gold[end_yaml + len(yaml_end_sep) :]
        gold = gold[start_yaml:end_yaml]
        frontmatter_gold = yaml.safe_load(gold)

    # compare the frontmatter (ignoring the order of keys and other elements)
    for key in frontmatter_gold:
        assert key in frontmatter, f"Key {key} not found in output"

        assert frontmatter[key] == frontmatter_gold[key], (
            f"Value for {key} does not match"
        )
    assert readme_output == gold_readme
    # ensure that the command line interface works as well
    command = f"{sys.executable} -m mteb create-model-results --model-name {model_name} --results-folder {output_folder.as_posix()} --output-path {output_path.as_posix()} --from-existing {existing_readme.as_posix()} --overwrite"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    assert result.returncode == 0, "Command failed"
