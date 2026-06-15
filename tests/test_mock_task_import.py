import mteb
from tests import MockSentenceTransformer, MockTask, test_tasks


def test_mteb_tests_import():
    assert isinstance(test_tasks, list)
    assert len(test_tasks) > 0

    model = MockSentenceTransformer()
    task = test_tasks[0]
    results = mteb.evaluate(model, task, cache=None)
    assert len(results.task_results) == 1


def test_mteb_mock_task():
    task = MockTask()
    assert task is not None
    assert len(task.tasks) > 0

    model = MockSentenceTransformer()
    subtask = task.tasks[0]
    results = mteb.evaluate(model, subtask, cache=None)
    assert len(results.task_results) == 1
