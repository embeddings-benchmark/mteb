from __future__ import annotations

from mteb.abstasks import AbsTaskSummarization
from mteb.abstasks.TaskMetadata import TaskMetadata


class SummEvalSwSummarization(AbsTaskSummarization):
    metadata = TaskMetadata(
        name="SummEvalSwSummarization",
        description="SummEval Dataset for Swahili",
        reference="https://huggingface.co/datasets/sproos/summeval-sw/",
        dataset={
            "path": "sproos/summeval-sw",
            "revision": "4eb1c5a78fa8f0e9236c1c969d399c3834944e1e"
        },
        type="Summarization",
        category="p2p",
        eval_splits=["train"],
        eval_langs=["swh-Latn"],
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
