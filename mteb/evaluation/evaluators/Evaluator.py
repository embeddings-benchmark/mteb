from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch

from mteb.encoder_interface import Encoder


class Evaluator(ABC):
    """Base class for all evaluators
    Extend this class and implement __call__ for custom evaluators.
    """

    def __init__(self, seed: int = 42, **kwargs: Any):
        self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

    @abstractmethod
    def __call__(self, model: Encoder, *, encode_kwargs: dict[str, Any] = {}):
        """This is called during training to evaluate the model.
        It returns scores.

        Args:
            model: the model to evaluate
            encode_kwargs: kwargs to pass to the model's encode method
        """
        pass
