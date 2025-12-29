"""tests for the MTEB CLI"""

import subprocess
import sys
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from mteb.cli.build_cli import (
    _available_benchmarks,
    _available_tasks,
    _create_meta,
    _leaderboard,
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


def test_leaderboard_help():
    """Test that leaderboard help command works."""
    command = [sys.executable, "-m", "mteb", "leaderboard", "--help"]
    result = subprocess.run(command, capture_output=True, text=True)

    assert result.returncode == 0, "Leaderboard help command failed"
    assert "--cache-path" in result.stdout, "--cache-path option not found in help"
    assert "--host" in result.stdout, "--host option not found in help"
    assert "--port" in result.stdout, "--port option not found in help"
    assert "--share" in result.stdout, "--share option not found in help"
    assert "Path to the cache folder containing model results" in result.stdout, (
        "Cache path description not found"
    )


def test_leaderboard_args(tmp_path: Path, monkeypatch):
    """Test leaderboard function with different arguments."""

    # Mock the gradio app to avoid actually launching it
    mock_app = MagicMock()
    mock_app.launch = MagicMock()

    with patch("mteb.leaderboard.get_leaderboard_app", return_value=mock_app):
        # Test with cache-path
        args = Namespace(
            cache_path=str(tmp_path / "custom_cache"),
            host="127.0.0.1",
            port=7860,
            share=False,
        )

        _leaderboard(args)

        # Verify launch was called with correct parameters
        mock_app.launch.assert_called_once_with(
            server_name="127.0.0.1",
            server_port=7860,
            share=False,
        )


def test_leaderboard_custom_cache_path(tmp_path: Path):
    """Test leaderboard with custom cache path."""

    custom_cache = tmp_path / "my_results"
    custom_cache.mkdir(exist_ok=True)

    mock_app = MagicMock()
    mock_app.launch = MagicMock()

    with patch(
        "mteb.leaderboard.get_leaderboard_app", return_value=mock_app
    ) as mock_get_app:
        with patch("mteb.cli.build_cli.ResultCache") as mock_result_cache:
            mock_cache_instance = MagicMock()
            mock_result_cache.return_value = mock_cache_instance

            args = Namespace(
                cache_path=str(custom_cache),
                host="localhost",
                port=8080,
                share=True,
            )

            _leaderboard(args)

            # Verify ResultCache was initialized with custom path
            mock_result_cache.assert_called_once_with(str(custom_cache))

            # Verify get_leaderboard_app was called with the cache instance
            mock_get_app.assert_called_once_with(mock_cache_instance)

            # Verify launch parameters
            mock_app.launch.assert_called_once_with(
                server_name="localhost",
                server_port=8080,
                share=True,
            )


def test_leaderboard_default_cache():
    """Test leaderboard with default cache path."""

    mock_app = MagicMock()
    mock_app.launch = MagicMock()

    with patch(
        "mteb.leaderboard.get_leaderboard_app", return_value=mock_app
    ) as mock_get_app:
        with patch("mteb.cli.build_cli.ResultCache") as mock_result_cache:
            mock_cache_instance = MagicMock()
            mock_result_cache.return_value = mock_cache_instance

            args = Namespace(
                cache_path=None,  # No cache path provided
                host="127.0.0.1",
                port=7860,
                share=False,
            )

            _leaderboard(args)

            # Verify ResultCache was initialized without arguments (default)
            mock_result_cache.assert_called_once_with()

            # Verify get_leaderboard_app was called with the cache instance
            mock_get_app.assert_called_once_with(mock_cache_instance)


def test_leaderboard_cli_integration():
    """Test the full CLI command integration."""
    # Test that the command is recognized by the CLI
    command = [sys.executable, "-m", "mteb", "--help"]
    result = subprocess.run(command, capture_output=True, text=True)

    assert result.returncode == 0
    assert "leaderboard" in result.stdout, "Leaderboard command not found in main help"
