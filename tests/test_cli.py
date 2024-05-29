"""tests for the MTEB CLI"""

import subprocess
from pathlib import Path


def test_available_tasks():
    command = "mteb --available_tasks"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    assert result.returncode == 0, "Command failed"
    assert (
        "Banking77Classification" in result.stdout
    ), "Sample task Banking77Classification task not found in available tasks"


def test_run_task(
    model_name: str = "average_word_embeddings_komninos",
    task_name="BornholmBitextMining",
):
    command = f"mteb -m {model_name} -t {task_name} --verbosity 3 --output_folder tests/results/test_model"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    assert result.returncode == 0, "Command failed"

    path = Path("tests/results/test_model")
    assert path.exists(), "Output folder not created"
    json_files = list(path.glob("*.json"))
    assert "model_meta.json" in [
        f.name for f in json_files
    ], "model_meta.json not found in output folder"
    assert f"{task_name}.json" in [
        f.name for f in json_files
    ], f"{task_name} not found in output folder"
