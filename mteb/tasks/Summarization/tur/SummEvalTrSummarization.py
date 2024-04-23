from __future__ import annotations

from mteb.abstasks import AbsTaskSummarization
from mteb.abstasks.TaskMetadata import TaskMetadata


class SummEvalTrSummarization(AbsTaskSummarization):
    metadata = TaskMetadata(
        name="SummEvalTrSummarization",
        description="SummEval Dataset for Swedish",
        reference="https://huggingface.co/datasets/sproos/summeval-tr/",
        dataset={
            "path": "sproos/summeval-tr",
            "revision": "d773054c462a22de9a3d4088ab49890bba06118f",
        },
        type="Summarization",
        category="p2p",
        eval_splits=["train"],
        eval_langs=["tur-Latn"],
        main_score="cosine_spearman",
        date=None,
        form=None,
        domains=None,
        task_subtypes=None,
        license=None,
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=None,
        text_creation=None,
        bibtex_citation=None,
        n_samples=None,
        avg_character_length=None,
    )

    @property
    def metadata_dict(self) -> dict[str, str]:
        metadata_dict = super().metadata_dict
        metadata_dict["min_score"] = 0
        metadata_dict["max_score"] = 5

        return metadata_dict
