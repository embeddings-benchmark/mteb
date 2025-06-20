from __future__ import annotations

import numpy as np
import pytest
import torch

import mteb
from mteb.evaluation.evaluators import (
    kNNClassificationEvaluator,
    kNNClassificationEvaluatorPytorch,
    logRegClassificationEvaluator,
)
from tests.test_benchmark.mock_models import MockNumpyEncoder

# Basic test data
SENTENCES_TRAIN_BINARY = [
    "this is a positive sentence",
    "another positive sentence",
    "this is a negative sentence",
    "another negative sentence",
]
Y_TRAIN_BINARY = np.array([1, 1, 0, 0])
SENTENCES_TEST_BINARY = [
    "a new positive sentence",
    "a new negative sentence",
]
Y_TEST_BINARY = np.array([1, 0])

SENTENCES_TRAIN_MULTICLASS = [
    "class 0 sentence 1",
    "class 0 sentence 2",
    "class 1 sentence 1",
    "class 1 sentence 2",
    "class 2 sentence 1",
    "class 2 sentence 2",
]
Y_TRAIN_MULTICLASS = np.array([0, 0, 1, 1, 2, 2])
SENTENCES_TEST_MULTICLASS = [
    "new class 0 sentence",
    "new class 1 sentence",
    "new class 2 sentence",
]
Y_TEST_MULTICLASS = np.array([0, 1, 2])

# For checking the cache
SENTENCES_TEST_CACHE = [
    "another new positive sentence",
    "another new negative sentence",
]
Y_TEST_CACHE = np.array([1, 0])


class TestKNNClassificationEvaluator:
    @pytest.fixture
    def model(self):
        return MockNumpyEncoder()

    @pytest.fixture
    def eval_binary(self):
        return kNNClassificationEvaluator(
            SENTENCES_TRAIN_BINARY,
            Y_TRAIN_BINARY,
            SENTENCES_TEST_BINARY,
            Y_TEST_BINARY,
            task_name="test_knn_binary",
        )

    @pytest.fixture
    def eval_multiclass(self):
        return kNNClassificationEvaluator(
            SENTENCES_TRAIN_MULTICLASS,
            Y_TRAIN_MULTICLASS,
            SENTENCES_TEST_MULTICLASS,
            Y_TEST_MULTICLASS,
            task_name="test_knn_multiclass",
        )

    @pytest.mark.parametrize(
        "evaluator_fixture, is_binary",
        [
            ("eval_binary", True),
            ("eval_multiclass", False),
        ],
    )
    def test_output_structure(self, evaluator_fixture, is_binary, model, request):
        evaluator = request.getfixturevalue(evaluator_fixture)
        scores, test_cache = evaluator(model)
        assert isinstance(scores, dict)
        assert isinstance(test_cache, np.ndarray)
        assert "accuracy" in scores
        assert "f1" in scores
        assert "accuracy_cosine" in scores
        assert "f1_cosine" in scores
        assert "accuracy_euclidean" in scores
        assert "f1_euclidean" in scores

        if is_binary:
            assert "ap" in scores
            assert "ap_cosine" in scores
            assert "ap_euclidean" in scores
        else:
            assert "ap" not in scores

    @pytest.mark.parametrize(
        "evaluator_fixture, is_binary",
        [
            ("eval_binary", True),
            ("eval_multiclass", False),
        ],
    )
    def test_score_ranges(self, evaluator_fixture, is_binary, model, request):
        evaluator = request.getfixturevalue(evaluator_fixture)
        scores, _ = evaluator(model)
        metrics_to_check = [
            "accuracy",
            "f1",
            "accuracy_cosine",
            "f1_cosine",
            "accuracy_euclidean",
            "f1_euclidean",
        ]
        if is_binary:
            metrics_to_check.extend(["ap", "ap_cosine", "ap_euclidean"])

        for metric in metrics_to_check:
            assert 0 <= scores[metric] <= 1

    def test_cache_usage_binary(self, model, eval_binary):
        _, test_cache_initial = eval_binary(model)
        eval_binary_cache_test = kNNClassificationEvaluator(
            SENTENCES_TRAIN_BINARY,
            Y_TRAIN_BINARY,
            SENTENCES_TEST_BINARY,
            Y_TEST_BINARY,
            task_name="test_knn_binary_cache",
        )
        scores_with_cache, test_cache_after_cache_usage = eval_binary_cache_test(
            model, test_cache=test_cache_initial
        )

        assert np.array_equal(test_cache_initial, test_cache_after_cache_usage)
        for metric in ["accuracy", "f1", "ap"]:
            assert 0 <= scores_with_cache[metric] <= 1

    @pytest.mark.parametrize(
        "train_sentences, train_labels, test_sentences, test_labels, task_name, limit, expected_train_len, expected_test_len, is_binary_after_limit, expected_exception",
        [
            # Ensure binary classification is still possible after limiting
            (
                SENTENCES_TRAIN_BINARY,
                Y_TRAIN_BINARY,
                SENTENCES_TEST_BINARY,
                Y_TEST_BINARY,
                "test_knn_limit_binary",
                3,
                3,
                2,
                True,
                None,
            ),
            # Test case where limiting results in a single class, expecting no AP and no exception
            (
                SENTENCES_TRAIN_BINARY,
                Y_TRAIN_BINARY,
                SENTENCES_TEST_BINARY,
                Y_TEST_BINARY,
                "test_knn_limit_not_binary",
                1,
                1,
                1,
                False,
                None,
            ),
        ],
    )
    def test_limit_parameter(
        self,
        model,
        train_sentences,
        train_labels,
        test_sentences,
        test_labels,
        task_name,
        limit,
        expected_train_len,
        expected_test_len,
        is_binary_after_limit,
        expected_exception,
    ):
        if expected_exception:
            with pytest.raises(expected_exception):
                eval_limited = kNNClassificationEvaluator(
                    train_sentences,
                    train_labels,
                    test_sentences,
                    test_labels,
                    task_name=task_name,
                    limit=limit,
                )
                eval_limited(model)
        else:
            eval_limited = kNNClassificationEvaluator(
                train_sentences,
                train_labels,
                test_sentences,
                test_labels,
                task_name=task_name,
                limit=limit,
            )
            assert len(eval_limited.sentences_train) == expected_train_len
            assert len(eval_limited.y_train) == expected_train_len
            assert len(eval_limited.sentences_test) == expected_test_len
            assert len(eval_limited.y_test) == expected_test_len

            scores, _ = eval_limited(model)
            assert "accuracy" in scores
            assert "f1" in scores
            if is_binary_after_limit:
                assert "ap" in scores
            else:
                assert "ap" not in scores


class LocalMockTorchEncoder(mteb.Encoder):
    def __init__(self):
        pass

    def encode(self, sentences, prompt_name: str | None = None, **kwargs):
        return torch.randn(len(sentences), 10)


class TestKNNClassificationEvaluatorPytorch:
    @pytest.fixture
    def model_pytorch(self):
        return LocalMockTorchEncoder()

    @pytest.fixture
    def eval_pytorch_binary(self):
        return kNNClassificationEvaluatorPytorch(
            SENTENCES_TRAIN_BINARY,
            Y_TRAIN_BINARY,
            SENTENCES_TEST_BINARY,
            Y_TEST_BINARY,
            task_name="test_knn_pytorch_binary",
        )

    @pytest.fixture
    def eval_pytorch_multiclass(self):
        return kNNClassificationEvaluatorPytorch(
            SENTENCES_TRAIN_MULTICLASS,
            Y_TRAIN_MULTICLASS,
            SENTENCES_TEST_MULTICLASS,
            Y_TEST_MULTICLASS,
            task_name="test_knn_pytorch_multiclass",
        )

    @pytest.mark.parametrize(
        "evaluator_fixture, is_binary",
        [
            ("eval_pytorch_binary", True),
            ("eval_pytorch_multiclass", False),
        ],
    )
    def test_output_structure(
        self, evaluator_fixture, is_binary, model_pytorch, request
    ):
        evaluator = request.getfixturevalue(evaluator_fixture)
        scores, test_cache = evaluator(model_pytorch)
        assert isinstance(scores, dict)
        assert isinstance(test_cache, torch.Tensor)
        assert "accuracy" in scores
        assert "f1" in scores
        assert "accuracy_cosine" in scores
        assert "f1_cosine" in scores
        assert "accuracy_euclidean" in scores
        assert "f1_euclidean" in scores
        assert "accuracy_dot" in scores
        assert "f1_dot" in scores

        if is_binary:
            assert "ap" in scores
            assert "ap_cosine" in scores
            assert "ap_euclidean" in scores
            assert "ap_dot" in scores
        else:
            assert "ap" not in scores

    @pytest.mark.parametrize(
        "evaluator_fixture, is_binary",
        [
            ("eval_pytorch_binary", True),
            ("eval_pytorch_multiclass", False),
        ],
    )
    def test_score_ranges(self, evaluator_fixture, is_binary, model_pytorch, request):
        evaluator = request.getfixturevalue(evaluator_fixture)
        scores, _ = evaluator(model_pytorch)
        metrics_to_check = [
            "accuracy",
            "f1",
            "accuracy_cosine",
            "f1_cosine",
            "accuracy_euclidean",
            "f1_euclidean",
            "accuracy_dot",
            "f1_dot",
        ]
        if is_binary:
            metrics_to_check.extend(["ap", "ap_cosine", "ap_euclidean", "ap_dot"])

        for metric_key in metrics_to_check:
            assert 0 <= scores[metric_key] <= 1

    def test_cache_usage_binary(self, model_pytorch, eval_pytorch_binary):
        _, test_cache_initial = eval_pytorch_binary(model_pytorch)
        eval_binary_cache_test = kNNClassificationEvaluatorPytorch(
            SENTENCES_TRAIN_BINARY,
            Y_TRAIN_BINARY,
            SENTENCES_TEST_BINARY,
            Y_TEST_BINARY,
            task_name="test_knn_pytorch_binary_cache",
        )
        scores_with_cache, test_cache_after_cache_usage = eval_binary_cache_test(
            model_pytorch, test_cache=test_cache_initial
        )

        assert torch.equal(test_cache_initial, test_cache_after_cache_usage)
        for metric_key in scores_with_cache.keys():
            if "accuracy" in metric_key or "f1" in metric_key or "ap" in metric_key:
                assert 0 <= scores_with_cache[metric_key] <= 1

    @pytest.mark.parametrize(
        "train_sentences, train_labels, test_sentences, test_labels, task_name, limit, expected_train_len, expected_test_len, is_binary_after_limit, expected_exception",
        [
            (
                SENTENCES_TRAIN_BINARY,
                Y_TRAIN_BINARY,
                SENTENCES_TEST_BINARY,
                Y_TEST_BINARY,
                "test_knn_pytorch_limit_binary",
                3,
                3,
                2,
                True,
                None,
            ),
            (
                SENTENCES_TRAIN_BINARY,
                Y_TRAIN_BINARY,
                SENTENCES_TEST_BINARY,
                Y_TEST_BINARY,
                "test_knn_pytorch_limit_not_binary",
                1,
                1,
                1,
                False,
                None,
            ),
        ],
    )
    def test_limit_parameter(
        self,
        model_pytorch,
        train_sentences,
        train_labels,
        test_sentences,
        test_labels,
        task_name,
        limit,
        expected_train_len,
        expected_test_len,
        is_binary_after_limit,
        expected_exception,
    ):
        if expected_exception:
            with pytest.raises(expected_exception):
                eval_limited = kNNClassificationEvaluatorPytorch(
                    train_sentences,
                    train_labels,
                    test_sentences,
                    test_labels,
                    task_name=task_name,
                    limit=limit,
                )
                eval_limited(model_pytorch)
        else:
            eval_limited = kNNClassificationEvaluatorPytorch(
                train_sentences,
                train_labels,
                test_sentences,
                test_labels,
                task_name=task_name,
                limit=limit,
            )
            assert len(eval_limited.sentences_train) == expected_train_len
            assert len(eval_limited.y_train) == expected_train_len
            assert len(eval_limited.sentences_test) == expected_test_len
            assert len(eval_limited.y_test) == expected_test_len

            scores, _ = eval_limited(model_pytorch)
            assert "accuracy" in scores
            assert "f1" in scores
            if is_binary_after_limit:
                assert "ap" in scores
            else:
                assert "ap" not in scores


class TestLogRegClassificationEvaluator:
    @pytest.fixture
    def model(self):
        return MockNumpyEncoder()

    @pytest.fixture
    def eval_logreg_binary(self):
        return logRegClassificationEvaluator(
            SENTENCES_TRAIN_BINARY,
            Y_TRAIN_BINARY,
            SENTENCES_TEST_BINARY,
            Y_TEST_BINARY,
            task_name="test_logreg_binary",
        )

    @pytest.fixture
    def eval_logreg_multiclass(self):
        return logRegClassificationEvaluator(
            SENTENCES_TRAIN_MULTICLASS,
            Y_TRAIN_MULTICLASS,
            SENTENCES_TEST_MULTICLASS,
            Y_TEST_MULTICLASS,
            task_name="test_logreg_multiclass",
        )

    @pytest.mark.parametrize(
        "evaluator_fixture, is_binary",
        [
            ("eval_logreg_binary", True),
            ("eval_logreg_multiclass", False),
        ],
    )
    def test_output_structure(self, evaluator_fixture, is_binary, model, request):
        evaluator = request.getfixturevalue(evaluator_fixture)
        scores, test_cache = evaluator(model)
        assert isinstance(scores, dict)
        assert isinstance(test_cache, np.ndarray)
        assert "accuracy" in scores
        assert "f1" in scores
        assert "f1_weighted" in scores

        if is_binary:
            assert "ap" in scores
            assert "ap_weighted" in scores
        else:
            assert "ap" not in scores

    @pytest.mark.parametrize(
        "evaluator_fixture, is_binary",
        [
            ("eval_logreg_binary", True),
            ("eval_logreg_multiclass", False),
        ],
    )
    def test_score_ranges(self, evaluator_fixture, is_binary, model, request):
        evaluator = request.getfixturevalue(evaluator_fixture)
        scores, _ = evaluator(model)
        metrics_to_check = ["accuracy", "f1", "f1_weighted"]
        if is_binary:
            metrics_to_check.extend(["ap", "ap_weighted"])

        for metric in metrics_to_check:
            assert 0 <= scores[metric] <= 1

    def test_cache_usage_binary(self, model, eval_logreg_binary):
        _, test_cache_initial = eval_logreg_binary(model)
        eval_binary_cache_test = logRegClassificationEvaluator(
            SENTENCES_TRAIN_BINARY,
            Y_TRAIN_BINARY,
            SENTENCES_TEST_BINARY,
            Y_TEST_BINARY,
            task_name="test_logreg_binary_cache",
        )
        scores_with_cache, test_cache_after_cache_usage = eval_binary_cache_test(
            model, test_cache=test_cache_initial
        )

        assert np.array_equal(test_cache_initial, test_cache_after_cache_usage)
        for metric in ["accuracy", "f1", "f1_weighted", "ap", "ap_weighted"]:
            assert 0 <= scores_with_cache[metric] <= 1

    @pytest.mark.parametrize(
        "train_sentences, train_labels, test_sentences, test_labels, task_name, limit, expected_train_len, expected_test_len, is_binary_after_limit, expected_exception",
        [
            # Ensure binary classification is still possible after limiting
            (
                SENTENCES_TRAIN_BINARY,
                Y_TRAIN_BINARY,
                SENTENCES_TEST_BINARY,
                Y_TEST_BINARY,
                "test_logreg_limit_binary",
                3,
                3,
                2,
                True,
                None,
            ),
            # Test case where limiting results in a single class, expecting ValueError
            (
                SENTENCES_TRAIN_BINARY,
                Y_TRAIN_BINARY,
                SENTENCES_TEST_BINARY,
                Y_TEST_BINARY,
                "test_logreg_limit_not_binary_train",
                1,
                1,
                1,
                False,
                ValueError,
            ),
        ],
    )
    def test_limit_parameter(
        self,
        model,
        train_sentences,
        train_labels,
        test_sentences,
        test_labels,
        task_name,
        limit,
        expected_train_len,
        expected_test_len,
        is_binary_after_limit,
        expected_exception,
    ):
        if expected_exception:
            with pytest.raises(expected_exception):
                eval_limited = logRegClassificationEvaluator(
                    train_sentences,
                    train_labels,
                    test_sentences,
                    test_labels,
                    task_name=task_name,
                    limit=limit,
                )
                eval_limited(model)
        else:
            eval_limited = logRegClassificationEvaluator(
                train_sentences,
                train_labels,
                test_sentences,
                test_labels,
                task_name=task_name,
                limit=limit,
            )
            assert len(eval_limited.sentences_train) == expected_train_len
            assert len(eval_limited.y_train) == expected_train_len
            assert len(eval_limited.sentences_test) == expected_test_len
            assert len(eval_limited.y_test) == expected_test_len

            scores, _ = eval_limited(model)
            assert "accuracy" in scores
            assert "f1" in scores
            assert "f1_weighted" in scores
            if is_binary_after_limit:
                assert "ap" in scores
                assert "ap_weighted" in scores
            else:
                assert "ap" not in scores
                assert "ap_weighted" not in scores
