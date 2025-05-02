from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from mteb.tasks.RTEB import RTEBLegalQuAD

if TYPE_CHECKING:
    from mteb.abstasks.AbsTaskRTEB import AbsTaskRTEB


@pytest.mark.parametrize("task", [RTEBLegalQuAD()])
def test_abstask_calculate_metadata_metrics(task: AbsTaskRTEB):
    task.calculate_metadata_metrics()
