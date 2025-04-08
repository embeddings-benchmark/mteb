from __future__ import annotations

from abc import ABC, abstractmethod
import time
import logging
from types import NoneType
from typing import Any, TYPE_CHECKING

import torch.nn as nn

if TYPE_CHECKING:
    from ebr.core.meta import ModelMeta


class EmbeddingModel(nn.Module, ABC):
    """Base class for embedding models.
    """

    def __init__(self,
        model_meta: ModelMeta,
        **kwargs
    ):
        super().__init__()
        self._model_meta = model_meta

    @abstractmethod
    def embed(self, data: Any, input_type: str) -> list[list[float]]:
        pass

    def forward(self, batch: dict[str, Any]) -> list[list[float]]:
        return self.embed(batch["text"], batch["input_type"][0])

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._model_meta, name)


class APIEmbeddingModel(EmbeddingModel):
    """Base class for API-based embedding models.
    """

    def __init__(self,
        model_meta: ModelMeta,
        api_key: str | None = None,
        num_retries: int | None = None,
        **kwargs
    ):
        super().__init__(model_meta, **kwargs)
        self._api_key = api_key
        assert num_retries is None or num_retries > 0, "num_retries must be a positive integer"
        self._num_retries = num_retries

    @property
    @abstractmethod
    def client(self) -> Any:
        pass

    def forward(self, batch: dict[str, Any]) -> list[list[float]]:
        num_tries = 0
        while not self._num_retries or num_tries < self._num_retries:
            try:
                num_tries += 1
                result = super().forward(batch)
                break
            except Exception as e:
                logging.error(e)
                if isinstance(e, type(self).rate_limit_error_type()):
                    time.sleep(60)
                elif isinstance(e, type(self).service_error_type()):
                    time.sleep(300)
                else:
                    raise e
        return result

    @property
    def api_key(self) -> str:
        return self._api_key

    @property
    def num_retries(self) -> int:
        return self._num_retries if self._num_retries else float("inf")

    @staticmethod
    def rate_limit_error_type() -> type:
        return NoneType

    @staticmethod
    def service_error_type() -> type:
        return NoneType
