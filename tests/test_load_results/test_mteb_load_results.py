import json
import os
from pathlib import Path

import pytest
from datasets import Dataset, DatasetDict

import mteb
from mteb import (
    MTEB, MTEBResults,
)
from tests.test_load_results.conftest import (
    MockEncoder,
    get_all_tasks_results,
)


def test_mteb_load_results():
    tests_path = Path(__file__).parent.parent

    os.environ["MTEB_CACHE"] = str(tests_path)

    results = mteb.load_results(download_latest=False)

    assert isinstance(results, dict)
    for model in results:
        assert isinstance(results[model], dict)
        for revision in results[model]:
            assert isinstance(results[model][revision], list)
            for result in results[model][revision]:
                assert isinstance(result, mteb.MTEBResults)

    known_model = "sentence-transformers/average_word_embeddings_levy_dependency"
    known_revision = "6d9c09a789ad5dd126b476323fccfeeafcd90509"
    assert known_model in results
    assert known_revision in results[known_model]


@pytest.mark.xfail
@pytest.mark.parametrize("task", MTEB().tasks_cls)
def test_load_results_main_score_in_real_results(task):
    """
    Test that main score is in real results scores with equal values
    """
    task_files = get_all_tasks_results()
    task_name = task.metadata.name
    result_files = task_files[task_name]
    for result_file in result_files:
        with open(result_file, "r") as f:
            result = json.load(f)
        assert "scores" in result.keys(), result_file + " not have 'scores'"
        for subset, subset_scores in result["scores"].items():
            assert isinstance(subset_scores, list), (
                result_file + " 'scores' is not list"
            )
            for subset_score in subset_scores:
                assert (
                    task.metadata.main_score in subset_score
                ), f"{result_file} not have {task.metadata.main_score} for task {task_name}"
                assert subset_score[task.metadata.main_score] == subset_score["main_score"], result_file


@pytest.mark.xfail(reason="Some classification datasets have additional keys")
@pytest.mark.parametrize(
    "task, dataset",
    [
        (
            "BitextMining",
            {"sentence1": ["test"] * 2, "sentence2": ["test"] * 2, "id": ["test"] * 2},
        ),
        (
            "Classification",
            {"text": ["test"] * 2, "label": [1, 0]},
        ),  # classification needs at least 2 classes
        ("Clustering", {"sentences": [["test"]] * 2, "labels": [[0], [1]]}),
        (
            "PairClassification",
            {
                "sentence1": [["test"]] * 2,
                "sentence2": [["test"]] * 2,
                "labels": [[1]] * 2,
            },
        ),
        (
            "Reranking",
            {
                "query": ["test"] * 2,
                "positive": [["test"]] * 2,
                "negative": [["test"]] * 2,
            },
        ),
        (
            "STS",
            {"sentence1": ["test"] * 2, "sentence2": ["test"] * 2, "score": [1] * 2},
        ),
        (
            "Summarization",
            {
                "text": ["text"],
                "human_summaries": [["text"]],
                "machine_summaries": [["text"]],
                "relevance": [[0.1]],
            },
        ),
    ],
)
def test_load_results_scores(task, dataset):
    """
    Test that all keys from actual task results presented in real task result
    """
    task_files = get_all_tasks_results()
    all_subclasses_classes = mteb.get_tasks(task_types=[task])
    example_task = all_subclasses_classes[0]
    example_task.is_multilingual = False
    example_task.data_loaded = True
    hf_dataset = Dataset.from_dict(dataset)
    example_task.dataset = DatasetDict(test=hf_dataset, train=hf_dataset)

    encoder = MockEncoder()
    res = example_task.evaluate(encoder, split="test")["default"]
    res_keys = sorted(list(res.keys()))
    for task_class in all_subclasses_classes:
        task_name = task_class.metadata.name
        result_files = task_files[task_name]
        for result_file in result_files:
            result = MTEBResults.from_disk(Path(result_file))

            for subset, subset_scores in result.scores.items():
                for subset_score in subset_scores:
                    subset_keys = list(subset_score.keys())
                    subset_keys.remove("hf_subset")
                    subset_keys.remove("languages")
                    subset_keys = sorted(subset_keys)
                    assert (
                        res_keys == subset_keys
                    ), f"{result_file} for task {task_name} keys not matching expected {res_keys}, actual {subset_keys}"


def test_load_results_scores_multilabel():
    """
    Test that all keys from actual task results presented in real task result
    """
    task_files = get_all_tasks_results()
    all_subclasses_classes = mteb.get_tasks(task_types=["MultilabelClassification"])
    example_task = all_subclasses_classes[0]
    example_task.is_multilingual = False
    example_task.data_loaded = True
    test_dataset = Dataset.from_dict(
        {"text": ["test", "test", "test"], "label": [[0, 1], [1, 0], [1, 0]]}
    )
    train_dataset = Dataset.from_dict(
        {"text": ["test"] * 100, "label": [[0, 1], [1, 0]] * 50}
    )
    example_task.dataset = DatasetDict(test=test_dataset, train=train_dataset)

    encoder = MockEncoder()
    res = example_task.evaluate(encoder, split="test")["default"]
    res_keys = sorted(list(res.keys()))
    for task_class in all_subclasses_classes:
        task_name = task_class.metadata.name
        result_files = task_files[task_name]
        for result_file in result_files:
            result = MTEBResults.from_disk(Path(result_file))
            for subset, subset_scores in result.scores.items():
                assert isinstance(subset_scores, list), (
                    result_file + " 'scores' is not list"
                )
                for subset_score in subset_scores:
                    subset_keys = list(subset_score.keys())
                    subset_keys.remove("hf_subset")
                    subset_keys.remove("languages")
                    subset_keys = sorted(subset_keys)
                    assert (
                        res_keys == subset_keys
                    ), f"{result_file} for task {task_name} keys not matching expected {res_keys}, actual {subset_keys}"


def test_load_results_scores_retrival():
    """
    Test that all keys from actual task results presented in real task result
    """
    task_files = get_all_tasks_results()
    all_subclasses_classes = mteb.get_tasks(task_types=["Retrieval"])
    example_task = all_subclasses_classes[0]
    example_task.is_multilingual = False
    example_task.data_loaded = True
    test_dataset = Dataset.from_dict(
        {
            "id": [1, 2],
            "context": ["test", "test"],
            "question": ["test", "test"],
            "answers": ["test", "test"],
        }
    )
    train_dataset = Dataset.from_dict(
        {
            "id": [1, 2],
            "context": ["test", "test"],
            "question": ["test", "test"],
            "answers": ["test", "test"],
        }
    )
    example_task.dataset = DatasetDict(test=test_dataset, train=train_dataset)
    example_task.corpus = {
        "test": {"document_one": {"_id": "d1", "title": "title", "text": "text"}}
    }
    example_task.queries = {"test": {"q1": ["turn1", "turn2", "turn3"]}}
    example_task.relevant_docs = {"test": {"q1": {"document_one": 1}}}

    encoder = MockEncoder()
    res = example_task.evaluate(encoder, split="test")["default"]
    res_keys = sorted(list(res.keys()))
    for task_class in all_subclasses_classes:
        task_name = task_class.metadata.name
        result_files = task_files[task_name]
        for result_file in result_files:
            result = MTEBResults.from_disk(Path(result_file))
            for subset, subset_scores in result.scores.items():
                assert isinstance(subset_scores, list), (
                    result_file + " 'scores' is not list"
                )
                for subset_score in subset_scores:
                    subset_keys = list(subset_score.keys())
                    subset_keys.remove("hf_subset")
                    subset_keys.remove("languages")
                    subset_keys = sorted(subset_keys)
                    assert (
                        res_keys == subset_keys
                    ), f"{result_file} for task {task_name} keys not matching expected {res_keys}, actual {subset_keys}"


def test_load_results_scores_gpu_speed():
    """
    Test that all keys from actual task results presented in real task result
    """
    task_files = get_all_tasks_results()
    pytest.importorskip("GPUtil")
    pytest.importorskip("psutil")

    all_subclasses_classes = mteb.get_tasks(task_types=["Speed"])
    example_task = all_subclasses_classes[0]
    encoder = MockEncoder()
    res = example_task.evaluate(encoder, split="test")["default"]
    res_keys = sorted(list(res.keys()))
    for task_class in all_subclasses_classes:
        task_name = task_class.metadata.name
        result_files = task_files[task_name]
        for result_file in result_files:
            result = MTEBResults.from_disk(Path(result_file))
            for subset, subset_scores in result.scores.items():
                assert isinstance(subset_scores, list), (
                    result_file + " 'scores' is not list"
                )
                for subset_score in subset_scores:
                    subset_keys = list(subset_score.keys())
                    subset_keys.remove("hf_subset")
                    subset_keys.remove("languages")
                    subset_keys = sorted(subset_keys)
                    assert (
                        res_keys == subset_keys
                    ), f"{result_file} for task {task_name} keys not matching expected {res_keys}, actual {subset_keys}"
