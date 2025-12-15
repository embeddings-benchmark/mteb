from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from typing import Any

from mteb.abstasks.abstask import _set_seed
from mteb.models import EncoderProtocol


class Evaluator(ABC):
    """Base class for all evaluators

    Extend this class and implement __call__ for custom evaluators.
    """

    def __init__(self, seed: int = 42, **kwargs: Any) -> None:
        self.seed = seed
        self.rng_state, self.np_rng = _set_seed(seed)

    @abstractmethod
    def __call__(
        self, model: EncoderProtocol, *, encode_kwargs: dict[str, Any]
    ) -> Mapping[str, float] | Iterable[Any]:
        """This is called during training to evaluate the model.

        It returns scores.

        Args:
            model: the model to evaluate
            encode_kwargs: kwargs to pass to the model's encode method
        """
        pass
