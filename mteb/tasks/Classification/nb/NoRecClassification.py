from __future__ import annotations

from mteb.abstasks import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class NoRecClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="NoRecClassification",
        description="A Norwegian dataset for sentiment classification on review",
        reference="https://aclanthology.org/L18-1661/",
        hf_hub_name="ScandEval/norec-mini",  # using the mini version to keep results ~comparable to the ScandEval benchmark
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["nb"],
        main_score="accuracy",
        revision="07b99ab3363c2e7f8f87015b01c21f4d9b917ce3",
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
    )

    @property
    def metadata_dict(self) -> dict[str, str]:
        return dict(self.metadata)
