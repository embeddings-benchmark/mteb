import mteb
from mteb.abstasks.mteb_result import MTEBResults


def test_mteb_results():
    """Test MTEBResults class (this is the same as the example in the docstring)"""

    class DummyTask:
        superseeded_by = "newer_task"
        metadata = mteb.TaskMetadata(
            name="dummy_task",
            description="dummy task for testing",
            dataset={"revision": "1.0", "name": "dummy_dataset"},
            type="Classification",
            category="p2p",
            eval_langs={
                "en-de": ["eng-Latn", "deu-Latn"],
                "en-fr": ["eng-Latn", "fra-Latn"],
            },
        )

    scores = {
        "train": {
            "en-de": {
                "main_score": 0.5,
                "evaluation_time": 100,
            },
            "en-fr": {
                "main_score": 0.6,
                "evaluation_time": 200,
            },
        },
    }

    mteb_results = MTEBResults(
        dataset_revision="1.0",
        task_name="dummy_task",
        mteb_version="1.0",
        scores=scores,
    )

    assert mteb_results.get_main_score() == 0.55
    assert mteb_results.get_main_score(languages=["fra"]) == 0.6
