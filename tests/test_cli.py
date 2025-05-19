"""tests for the MTEB CLI"""

from __future__ import annotations

import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pytest
import yaml

from mteb.cli import create_meta, run


def test_available_tasks():
    command = f"{sys.executable} -m mteb available_tasks"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    assert result.returncode == 0, "Command failed"
    assert "Banking77Classification" in result.stdout, (
        "Sample task Banking77Classification task not found in available tasks"
    )


def test_available_benchmarks():
    command = f"{sys.executable} -m mteb available_benchmarks"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    assert result.returncode == 0, "Command failed"
    assert "MTEB(eng, v1)" in result.stdout, (
        "Sample benchmark MTEB(eng, v1) task not found in available benchmarks"
    )


run_task_fixures = [
    (
        "sentence-transformers/average_word_embeddings_komninos",
        "BornholmBitextMining",
        "21eec43590414cb8e3a6f654857abed0483ae36e",
    ),
    (
        "intfloat/multilingual-e5-small",
        "BornholmBitextMining",
        "fd1525a9fd15316a2d503bf26ab031a61d056e98",
    ),
]


@pytest.mark.parametrize("model_name,task_name,model_revision", run_task_fixures)
def test_run_task(
    model_name: str,
    task_name: str,
    model_revision: str,
):
    args = Namespace(
        model=model_name,
        tasks=[task_name],
        model_revision=model_revision,
        output_folder="tests/results/test_model",
        verbosity=3,
        device=None,
        categories=None,
        task_types=None,
        languages=None,
        batch_size=None,
        co2_tracker=None,
        overwrite=True,
        eval_splits=None,
        benchmarks=None,
    )

    run(args)

    model_name_as_path = model_name.replace("/", "__").replace(" ", "_")
    results_path = Path(
        f"tests/results/test_model/{model_name_as_path}/{model_revision}"
    )
    assert results_path.exists(), "Output folder not created"
    assert "model_meta.json" in [f.name for f in list(results_path.glob("*.json"))], (
        "model_meta.json not found in output folder"
    )
    assert f"{task_name}.json" in [f.name for f in list(results_path.glob("*.json"))], (
        f"{task_name} not found in output folder"
    )


def test_create_meta():
    """Test create_meta function directly as well as through the command line interface"""
    test_folder = Path(__file__).parent
    output_folder = test_folder / "create_meta"
    results = (
        output_folder / "all-MiniLM-L6-v2" / "8b3219a92973c328a8e22fadcfa821b5dc75636a"
    )
    output_path = output_folder / "model_card.md"

    args = Namespace(
        results_folder=results,
        output_path=output_path,
        overwrite=True,
        from_existing=None,
    )

    create_meta(args)

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
    command = f"{sys.executable} -m mteb create_meta --results_folder {results} --output_path {output_path} --overwrite"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    assert result.returncode == 0, "Command failed"


@pytest.mark.parametrize(
    "existing_readme_name, gold_readme_name",
    [
        ("existing_readme.md", "model_card_gold_existing.md"),
        ("model_card_without_frontmatter.md", "model_card_gold_without_frontmatter.md"),
    ],
)
def test_create_meta_from_existing(existing_readme_name: str, gold_readme_name: str):
    """Test create_meta function directly as well as through the command line interface"""
    test_folder = Path(__file__).parent
    output_folder = test_folder / "create_meta"
    results = (
        output_folder / "all-MiniLM-L6-v2" / "8b3219a92973c328a8e22fadcfa821b5dc75636a"
    )
    output_path = output_folder / "model_card.md"
    existing_readme = output_folder / existing_readme_name

    args = Namespace(
        results_folder=results,
        output_path=output_path,
        overwrite=True,
        from_existing=str(existing_readme),
    )

    create_meta(args)

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
    command = f"{sys.executable} -m mteb create_meta --results_folder {results} --output_path {output_path} --from_existing {existing_readme} --overwrite"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    assert result.returncode == 0, "Command failed"


def test_save_predictions():
    command = f"{sys.executable} -m mteb run -m sentence-transformers/average_word_embeddings_komninos -t NFCorpus --output_folder tests/results --save_predictions"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    assert result.returncode == 0, "Command failed"
    test_folder = Path(__file__).parent
    results_path = test_folder / "results" / "NFCorpus_default_predictions.json"
    assert results_path.exists(), "Predictions file not created"
