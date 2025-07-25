from __future__ import annotations

import logging
from typing import Any, Protocol

import numpy as np
from scipy.stats import kendalltau
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from mteb.encoder_interface import Encoder

from .Evaluator import Evaluator

logger = logging.getLogger(__name__)


class SklearnRegressorModel(Protocol):
    def fit(self, X, y, sample_weight=None): ...
    def predict(self, X): ...


class LinearRegressionEvaluator(Evaluator):
    def __init__(
        self,
        sentences_train,
        y_train,
        sentences_test,
        y_test,
        task_name: str,
        hf_split: str,
        hf_subset: str,
        regressor: SklearnRegressorModel,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sentences_train = sentences_train
        self.y_train = y_train
        self.sentences_test = sentences_test
        self.y_test = y_test

        self.hf_split = hf_split
        self.hf_subset = hf_subset
        self.task_name = task_name
        self.regressor = regressor

    def __call__(
        self,
        model: Encoder,
        *,
        encode_kwargs: dict[str, Any] = {},
        test_cache: np.ndarray | None = None,
    ) -> tuple[dict[str, float], np.ndarray]:
        scores = {}
        X_train = model.encode(
            self.sentences_train,
            model=model,
            task_name=self.task_name,
            hf_split="train",
            hf_subset=self.hf_subset,
            **encode_kwargs,
        )
        if test_cache is None:
            X_test = model.encode(
                self.sentences_test,
                model=model,
                task_name=self.task_name,
                hf_split=self.hf_split,
                hf_subset=self.hf_subset,
                **encode_kwargs,
            )
            test_cache = X_test

        self.regressor.fit(X_train, self.y_train)
        y_pred = self.regressor.predict(test_cache)

        scores["mae"] = mean_absolute_error(self.y_test, y_pred)
        scores["mse"] = mean_squared_error(self.y_test, y_pred)
        scores["rmse"] = np.sqrt(scores["mse"])
        scores["r2"] = r2_score(self.y_test, y_pred)
        scores["kendalltau"] = kendalltau(self.y_test, y_pred).statistic

        return scores, test_cache
