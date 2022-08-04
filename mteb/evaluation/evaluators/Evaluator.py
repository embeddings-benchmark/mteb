from abc import ABC, abstractmethod
import random

import numpy as np
import torch

class Evaluator(ABC):
    """
    Base class for all evaluators
    Extend this class and implement __call__ for custom evaluators.
    """
    def __init__(self, **kwargs):
        self.seed = kwargs.get("seed", 42)

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

    @abstractmethod
    def __call__(self, model):
        """
        This is called during training to evaluate the model.
        It returns scores.

        Parameters
        ----------
        model:
            the model to evaluate
        """
        pass
