import glob
import json
import os
from importlib.metadata import version
from pathlib import Path

import pytest
from packaging.version import Version

import mteb
from mteb import AbsTask
from mteb.load_results.mteb_results import MTEBResults

tests_folder = Path(__file__).parent


class DummyTask(AbsTask):
    superseded_by = "newer_task"
    metadata = mteb.TaskMetadata(
        name="dummy_task",
        description="dummy task for testing",
        dataset={"revision": "1.0", "path": "dummy_dataset"},
        type="Classification",
        category="p2p",
        eval_langs={
            "en-de": ["eng-Latn", "deu-Latn"],
            "en-fr": ["eng-Latn", "fra-Latn"],
        },
        main_score="main_score",
        eval_splits=["train"],
        domains=[],
        text_creation="created",
        reference="https://www.noreference.com",
        date=("2024-05-02", "2024-05-03"),
        form=[],
        task_subtypes=[],
        license="mit",
        socioeconomic_status="mixed",
        annotations_creators="derived",
        dialect=[],
        bibtex_citation="",
        avg_character_length={},
        n_samples={},
    )

    def evaluate(self, model, split: str = "test"):
        pass

    def _evaluate_subset(self, **kwargs):
        pass


def test_mteb_results():
    """Test MTEBResults class (this is the same as the example in the docstring)"""
    scores = {
        "train": {
            "en-de": {
                "main_score": 0.5,
            },
            "en-fr": {
                "main_score": 0.6,
            },
        },
    }

    evaluation_time = 100

    mteb_results = MTEBResults.from_task_results(
        task=DummyTask(), scores=scores, evaluation_time=evaluation_time
    )

    assert mteb_results.get_score() == 0.55
    assert mteb_results.get_score(languages=["eng"]) == 0.55
    assert mteb_results.get_score(languages=["fra"]) == 0.6
    dict_repr = {
        "dataset_revision": "1.0",
        "task_name": "dummy_task",
        "mteb_version": version("mteb"),
        "evaluation_time": 100,
        "kg_co2_emissions": None,
        "scores": {
            "train": [
                {
                    "main_score": 0.5,
                    "hf_subset": "en-de",
                    "languages": ["eng-Latn", "deu-Latn"],
                },
                {
                    "main_score": 0.6,
                    "hf_subset": "en-fr",
                    "languages": ["eng-Latn", "fra-Latn"],
                },
            ]
        },
    }
    assert mteb_results.to_dict() == dict_repr


@pytest.mark.parametrize(
    "path", list((tests_folder / "historic_results").glob("*.json"))
)
def test_mteb_results_from_historic(path: Path):
    mteb_result = MTEBResults.from_disk(path, load_historic_data=True)
    assert isinstance(mteb_result, MTEBResults)


def get_json_files(results_path):
    """Get all *.json files in the results folder, including nested directories, excluding model_meta files."""
    all_files = glob.glob(os.path.join(results_path, "**", "*.json"), recursive=True)
    return [x for x in all_files if "model_meta" not in x]


def get_version_from_json(json_path):
    # Placeholder for reading the version from the JSON file
    # In real implementation, you would read and parse the JSON to get the version
    with open(json_path) as f:
        data = json.load(f)
    return data.get("mteb_version")


@pytest.fixture
def results_path():
    return "results"


@pytest.mark.parametrize("json_path", get_json_files("results"))
def test_revision_layer(json_path, results_path):
    relative_path = os.path.relpath(json_path, results_path)
    parts = relative_path.split(os.sep)

    # Get the version from the JSON file
    json_version = get_version_from_json(json_path)

    # Check if the version is higher than 1.12.6
    if Version(json_version) > Version("1.12.6"):
        # If version is higher than 1.12.6, there must be a revision layer
        assert (
            len(parts) == 3
        ), f"Path '{relative_path}' does not have a revision layer but version is {json_version}"
    else:
        # If version is 1.12.6 or lower, the revision layer might not be present
        assert len(parts) in [
            2,
            3,
        ], f"Unexpected path structure for version {json_version}: '{relative_path}'"


if __name__ == "__main__":
    pytest.main([__file__])
