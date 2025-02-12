from __future__ import annotations

from mteb.abstasks.AbsTaskSTS import AbsTaskSTS
from mteb.abstasks.TaskMetadata import TaskMetadata


class BQCorpus(AbsTaskSTS):
    metadata = TaskMetadata(
        name="BQCorpus",
        description="Bank Question Corpus: A Chinese corpus for sentence semantic equivalence identification (SSEI).",
        reference="https://aclanthology.org/D18-1536/",
        dataset={
            "path": "FinanceMTEB/bq_corpus",
            "revision": "24a4a7cfa6fb8ab07f214809fefed1cd4e8250cb",
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
