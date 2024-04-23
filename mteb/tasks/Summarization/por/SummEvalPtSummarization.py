from __future__ import annotations

from mteb.abstasks import AbsTaskSummarization
from mteb.abstasks.TaskMetadata import TaskMetadata


class SummEvalPtSummarization(AbsTaskSummarization):
    metadata = TaskMetadata(
        name="SummEvalPtSummarization",
        description="Portugese Summarization Dataset",
        reference="https://huggingface.co/datasets/mteb-pt/summeval",
        dataset={
            "path": "mteb-pt/summeval",
            "revision": "b57d536d89da6fc42b3ca40645bb813f538b9ede",
        },
        type="Summarization",
        category="p2p",
        eval_splits=["test"],
        eval_langs=["por-Latn"],
        main_score="cosine_spearman",
        date=('2022-07-20' - '2024-03-27'),
        form=["written"],
        domains=None,
        task_subtypes=None,
        license="Not specified",
        socioeconomic_status="high",
        annotations_creators="derived",
        dialect=[],
        text_creation="found",
        bibtex_citation=None,
        n_samples=100,
        avg_character_length=None,
    )

    @property
    def metadata_dict(self) -> dict[str, str]:
        metadata_dict = super().metadata_dict
        metadata_dict["min_score"] = 0
        metadata_dict["max_score"] = 5

        return metadata_dict
