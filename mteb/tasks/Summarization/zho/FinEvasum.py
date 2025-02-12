from __future__ import annotations

from mteb.abstasks.AbsTaskSTS import AbsTaskSTS
from mteb.abstasks.TaskMetadata import TaskMetadata


class FinEvasum(AbsTaskSTS):
    metadata = TaskMetadata(
        name="FinEvasum",
        description=" A financial news summarization dataset.",
        reference="https://github.com/alipay/financial_evaluation_dataset/",
        dataset={
            "path": "FinanceMTEB/FinEvaSum",
            "revision": "094d9c825e66d53c17cd3c3bb258523b8a12af15",
        },
        type="Summarization",
        category="p2p",
        eval_splits=["test"],
        eval_langs=["cmn-Hans"],
        main_score="cosine_spearman",
    )
    sentence_1_column = "text"
    sentence_2_column = "summary"

    @property
    def metadata_dict(self) -> dict[str, str]:
        metadata_dict = super().metadata_dict
        metadata_dict["min_score"] = 0
        metadata_dict["max_score"] = 1
        return metadata_dict
