from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from mteb.abstasks.abstask import _set_seed

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from mteb.models import EncoderProtocol
    from mteb.types import EncodeKwargs


class Evaluator(ABC):
    """Base class for all evaluators

    Extend this class and implement __call__ for custom evaluators.
    """

    def __init__(self, seed: int = 42, **kwargs: Any) -> None:
        self.seed = seed
        self.rng_state, self.np_rng = _set_seed(seed)

    @abstractmethod
    def __call__(
        self, model: EncoderProtocol, *, encode_kwargs: EncodeKwargs, num_proc: int = 1
    ) -> Mapping[str, float] | Iterable[Any]:
        """This is called during training to evaluate the model.

        It returns scores.

        Args:
            model: the model to evaluate
            encode_kwargs: kwargs to pass to the model's encode method
            num_proc: number of processes to use for data loading
        """
        pass
