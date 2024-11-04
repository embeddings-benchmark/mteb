from __future__ import annotations

from mteb.abstasks.AbsTaskSTS import AbsTaskSTS
from mteb.abstasks.TaskMetadata import TaskMetadata


class FinSTS(AbsTaskSTS):
    metadata = TaskMetadata(
        name="FinSTS",
        description="Detecting Subtle Semantic Shifts in Financial Narratives.",
        reference="https://arxiv.org/pdf/2403.14341",
        dataset={
            "path": "FinanceMTEB/FinSTS",
            "revision": "09e270b1afe87a65dd41c6292e3c8905220bc290",
        },
        type="STS",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="cosine_spearman",
    )

    @property
    def metadata_dict(self) -> dict[str, str]:
        metadata_dict = super().metadata_dict
        metadata_dict["min_score"] = 0
        metadata_dict["max_score"] = 1
        return metadata_dict
