from __future__ import annotations

from mteb.abstasks.AbsTaskSTS import AbsTaskSTS
from mteb.abstasks.TaskMetadata import TaskMetadata


class FiNNAsum(AbsTaskSTS):
    metadata = TaskMetadata(
        name="FiNNAsum",
        description=" A financial news summarization dataset.",
        reference="https://github.com/ssymmetry/BBT-FinCUGE-Applications/blob/main/FinCUGE_Publish/finna/train_list.json",
        dataset={
            "path": "FinanceMTEB/FinNA",
            "revision": "6c7e4c5807dbe5c53f9d4d445a9f707607f7e286",
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
