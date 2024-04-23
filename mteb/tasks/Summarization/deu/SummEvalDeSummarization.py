from __future__ import annotations

from mteb.abstasks import AbsTaskSummarization
from mteb.abstasks.TaskMetadata import TaskMetadata


class SummEvalDeSummarization(AbsTaskSummarization):
    metadata = TaskMetadata(
        name="SummEvalDeSummarization",
        description="SummEval Dataset for German",
        reference="https://huggingface.co/datasets/sproos/summeval-de/",
        dataset={
            "path": "sproos/summeval-de",
            "revision": "d25b21e407a2d22c96681fe3ae700eb2cc0c9bb9",
        },
        type="Summarization",
        category="p2p",
        eval_splits=["train"],
        eval_langs=["deu-Latn"],
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
