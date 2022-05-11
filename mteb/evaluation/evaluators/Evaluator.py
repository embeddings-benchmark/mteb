from abc import ABC, abstractmethod


class Evaluator(ABC):
    """
    Base class for all evaluators
    Extend this class and implement __call__ for custom evaluators.
    """

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
