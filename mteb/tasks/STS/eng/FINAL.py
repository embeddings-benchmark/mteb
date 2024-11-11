from __future__ import annotations

from mteb.abstasks.AbsTaskSTS import AbsTaskSTS
from mteb.abstasks.TaskMetadata import TaskMetadata


class FINAL(AbsTaskSTS):
    metadata = TaskMetadata(
        name="FINAL",
        description="A dataset for discovering financial signals in narrative financial reports.",
        reference="https://aclanthology.org/2023.acl-long.800.pdf",
        dataset={
            "path": "FinanceMTEB/Final",
            "revision": "00506d0f1853ebe9fcc5112fb36a4cc4cc521695",
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
