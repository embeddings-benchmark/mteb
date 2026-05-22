from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Protocol

from .evaluator import Evaluator

if TYPE_CHECKING:
    import numpy as np
    from datasets import Dataset
    from numpy.typing import NDArray
    from typing_extensions import Self

    from mteb.types import Array

logger = logging.getLogger(__name__)


class SklearnModelProtocol(Protocol):
    def fit(
        self, X: Array, y: NDArray[np.integer | np.floating] | list[int | float]
    ) -> None: ...
    def predict(self, X: Array) -> NDArray[np.integer | np.floating]: ...
    def get_params(self) -> dict[str, Any]: ...
    def set_params(self, random_state: int, **kwargs: dict[str, Any]) -> Self: ...
    def score(
        self, X: Array, y: NDArray[np.integer | np.floating] | list[int | float]
    ) -> float: ...


class SklearnEvaluator(Evaluator):
    def __init__(
        self,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        *,
        label_column_name: str,
        evaluator_model: SklearnModelProtocol,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.label_column_name = label_column_name
        self.evaluator_model = evaluator_model

    def __call__(  # type: ignore[override]
        self,
        train_embeddings: Array,
        test_embeddings: Array,
    ) -> NDArray[np.integer | np.floating]:
        y_train = self.train_dataset[self.label_column_name]
        logger.info("Running - Fitting classifier...")
        self.evaluator_model.fit(train_embeddings, y_train)
        return self.evaluator_model.predict(test_embeddings)
