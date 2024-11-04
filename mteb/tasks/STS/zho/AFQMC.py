from __future__ import annotations

from mteb.abstasks.AbsTaskSTS import AbsTaskSTS
from mteb.abstasks.TaskMetadata import TaskMetadata


class AFQMC(AbsTaskSTS):
    metadata = TaskMetadata(
        name="AFQMC",
        description="Ant Financial Question Matching Corpus.",
        reference="https://tianchi.aliyun.com/dataset/106411",
        dataset={
            "path": "FinanceMTEB/AFQMC",
            "revision": "f730edcecdc5d52dd71b5446c08976d3fd2d58ab",
        },
        type="STS",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["cmn-Hans"],
        main_score="cosine_spearman",
    )

    @property
    def metadata_dict(self) -> dict[str, str]:
        metadata_dict = super().metadata_dict
        metadata_dict["min_score"] = 0
        metadata_dict["max_score"] = 1
        return metadata_dict
