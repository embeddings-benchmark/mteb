from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


def get_model_names() -> list[str]:
    model_names = []
    model_names_file_path = (
        Path(__file__).parent.parent.parent / "scripts" / "model_names.txt"
    )
    if model_names_file_path.exists():
        with model_names_file_path.open("r") as f:
            model_names = f.read().strip()
    return model_names


model_names = get_model_names()


@pytest.mark.skipif(not model_names, reason="No updates to models.")
def test_model_loading():
    model_loading_file_path = (
        Path(__file__).parent.parent.parent / "scripts" / "model_loading.py"
    )
    result = subprocess.run(
        ["python", model_loading_file_path, "--model_name", model_names],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
