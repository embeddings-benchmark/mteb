from __future__ import annotations

import logging
from .AbsTask import AbsTask
logger = logging.getLogger(__name__)


class AbsTaskAbstention(AbsTask):
    """Abstract class for Abstention experiments.

    Abstention tasks are multi-inheritance tasks that inherit from a base task as well as the abstention task.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _evaluate_monolingual(self, *args, **kwargs):
        raise NotImplementedError("evaluate() must be re-implemented in the concrete task.")
