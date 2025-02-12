from __future__ import annotations

from mteb.abstasks.AbsTaskSTS import AbsTaskSTS
from mteb.abstasks.TaskMetadata import TaskMetadata


class FinEvaHeadlinesum(AbsTaskSTS):
    metadata = TaskMetadata(
        name="FinEvaHeadlinesum",
        description=" A financial news summarization dataset.",
        reference="https://github.com/alipay/financial_evaluation_dataset/",
        dataset={
            "path": "FinanceMTEB/FinEvaHeadline",
            "revision": "4c72b5793c541449d3289e4d86f54883fa633313",
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
