from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class AngryTweetsClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="AngryTweetsClassification",
        hf_hub_name="DDSC/angry-tweets",
        description="A sentiment dataset with 3 classes (positiv, negativ, neutral) for Danish tweets",
        reference="https://aclanthology.org/2021.nodalida-main.53/",
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["da"],
        main_score="accuracy",
        revision="20b0e6081892e78179356fada741b7afa381443d",
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
        metadata_dict = dict(self.metadata)
        metadata_dict["n_experiments"] = 10
        metadata_dict["samples_per_label"] = 16
        return metadata_dict
