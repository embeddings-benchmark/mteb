from __future__ import annotations

from mteb.abstasks.AbsTaskSTS import AbsTaskSTS
from mteb.abstasks.TaskMetadata import TaskMetadata


class FNS2022sum(AbsTaskSTS):
    metadata = TaskMetadata(
        name="FNS2022sum",
        description="Financial Narrative Summarisation for 10K. .",
        reference="https://wp.lancs.ac.uk/cfie/fns2022/",
        dataset={
            "path": "FinanceMTEB/FNS",
            "revision": "3a6eaec26efc16f5668d0c1511b4309152998466",
        },
        type="Summarization",
        category="p2p",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
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
