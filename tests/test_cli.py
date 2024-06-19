"""tests for the MTEB CLI"""

import subprocess
from argparse import Namespace
from pathlib import Path

import pytest
import yaml

from mteb.cli import create_meta


def test_available_tasks():
    command = "mteb available_tasks"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    assert result.returncode == 0, "Command failed"
    assert (
        "Banking77Classification" in result.stdout
    ), "Sample task Banking77Classification task not found in available tasks"


run_task_fixures = [
    (
        "average_word_embeddings_komninos",
        "BornholmBitextMining",
        "21eec43590414cb8e3a6f654857abed0483ae36e",
    ),
    (
        "intfloat/multilingual-e5-small",
        "BornholmBitextMining",
        "e4ce9877abf3edfe10b0d82785e83bdcb973e22e",
    ),
]


@pytest.mark.parametrize("model_name,task_name,model_revision", run_task_fixures)
def test_run_task(
    model_name: str,
    task_name: str,
    model_revision: str,
):
    command = f"mteb run -m {model_name} -t {task_name} --verbosity 3 --output_folder tests/results/test_model --model_revision {model_revision}"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    assert result.returncode == 0, "Command failed"

    model_name_as_path = model_name.replace("/", "__").replace(" ", "_")
    results_path = Path(
        f"tests/results/test_model/{model_name_as_path}/{model_revision}"
    )
    assert results_path.exists(), "Output folder not created"
    assert "model_meta.json" in [
        f.name for f in list(results_path.glob("*.json"))
    ], "model_meta.json not found in output folder"
    assert f"{task_name}.json" in [
        f.name for f in list(results_path.glob("*.json"))
    ], f"{task_name} not found in output folder"


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

        assert (
            frontmatter[key] == frontmatter_gold[key]
        ), f"Value for {key} does not match"

    # ensure that the command line interface works as well
    command = f"mteb create_meta --results_folder {results} --output_path {output_path} --overwrite"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    assert result.returncode == 0, "Command failed"
