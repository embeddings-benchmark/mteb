from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy.stats import kendalltau
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from mteb.encoder_interface import Encoder

from .Evaluator import Evaluator

logger = logging.getLogger(__name__)


class LinearRegressionEvaluator(Evaluator):
    def __init__(
        self,
        sentences_train,
        y_train,
        sentences_test,
        y_test,
        task_name: str | None = None,
        fit_intercept: bool = True,
        encode_kwargs: dict[str, Any] = {},
        limit: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if limit is not None:
            sentences_train = sentences_train[:limit]
            y_train = y_train[:limit]
            sentences_test = sentences_test[:limit]
            y_test = y_test[:limit]
        self.sentences_train = sentences_train
        self.y_train = y_train
        self.sentences_test = sentences_test
        self.y_test = y_test

        self.task_name = task_name
        self.encode_kwargs = encode_kwargs

        if "batch_size" not in self.encode_kwargs:
            self.encode_kwargs["batch_size"] = 32

        self.fit_intercept = fit_intercept

    def __call__(self, model: Encoder) -> dict[str, float]:
        scores = {}
        X_train = model.encode(
            self.sentences_train,
            model=model,
            task_name=self.task_name,
            **self.encode_kwargs,
        )
        X_test = model.encode(
            self.sentences_test,
            model=model,
            task_name=self.task_name,
            **self.encode_kwargs,
        )

        linear_regression = LinearRegression(fit_intercept=self.fit_intercept)
        linear_regression.fit(X_train, self.y_train)
        y_pred = linear_regression.predict(X_test)

        scores["mae"] = mean_absolute_error(self.y_test, y_pred)
        scores["mse"] = mean_squared_error(self.y_test, y_pred)
        scores["rmse"] = np.sqrt(scores["mse"])
        scores["r2"] = r2_score(self.y_test, y_pred)
        scores["kendalltau"] = kendalltau(self.y_test, y_pred).statistic

        return scores
