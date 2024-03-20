from mteb.abstasks import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class NorwegianParliamentClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="NorwegianParliamentClassification",
        description="Norwegian parliament speeches annotated for sentiment",
        reference="https://huggingface.co/datasets/NbAiLab/norwegian_parliament",
        hf_hub_name="NbAiLab/norwegian_parliament",
        type="Classification",
        category="s2s",
        eval_splits=["test", "validation"],
        eval_langs=["nb"],  # assumed to be bokmål
        main_score="accuracy",
        revision="f7393532774c66312378d30b197610b43d751972",
        date=None,
        form=None,
        domains=None,
        task_subtypes=None,
        license="",
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=None,
        text_creation=None,
        bibtex_citation=None,
    )

    @property
    def metadata_dict(self) -> dict[str, str]:
        return dict(self.metadata)
