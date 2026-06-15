import mteb
from mteb.tests import test_tasks
from tests.mock_models import MockSentenceTransformer


def test_mteb_tests_import():
    assert isinstance(test_tasks, list)
    assert len(test_tasks) > 0

    model = MockSentenceTransformer()
    task = test_tasks[0]
    results = mteb.evaluate(model, task, cache=None)
    assert len(results.task_results) == 1
