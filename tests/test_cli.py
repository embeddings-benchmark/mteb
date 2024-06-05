"""tests for the MTEB CLI"""

import subprocess
from pathlib import Path


def test_available_tasks():
    command = "mteb available_tasks"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    assert result.returncode == 0, "Command failed"
    assert (
        "Banking77Classification" in result.stdout
    ), "Sample task Banking77Classification task not found in available tasks"


def test_run_task(
    model_name: str = "average_word_embeddings_komninos",
    task_name="BornholmBitextMining",
    model_revision="21eec43590414cb8e3a6f654857abed0483ae36e",
):
    command = f"mteb run -m {model_name} -t {task_name} --verbosity 3 --output_folder tests/results/test_model --model_revision {model_revision}"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    assert result.returncode == 0, "Command failed"

    results_path = Path(
        f"tests/results/test_model/average_word_embeddings_komninos/{model_revision}"
    )
    assert results_path.exists(), "Output folder not created"
    assert "model_meta.json" in [
        f.name for f in list(results_path.glob("*.json"))
    ], "model_meta.json not found in output folder"
    assert f"{task_name}.json" in [
        f.name for f in list(results_path.glob("*.json"))
    ], f"{task_name} not found in output folder"


def test_create_meta():
    test_folder = Path(__file__).parent
    output_folder = test_folder / "create_meta"
    results = (
        output_folder / "all-MiniLM-L6-v2" / "8b3219a92973c328a8e22fadcfa821b5dc75636a"
    )
    output_path = output_folder / "model_card.md"
    command = f"mteb create_meta --results_folder {results} --output_path {output_path} --overwrite"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    assert result.returncode == 0, "Command failed"

    assert output_path.exists(), "Output file not created"

    with output_path.open("r") as f:
        meta = f.read()

    with (output_folder / "model_card_gold.md").open("r") as f:
        gold = f.read()

    assert meta == gold, "Output does not match gold"
