import datasets

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class DKHateClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="DKHateClassification",
        hf_hub_name="DDSC/dkhate",
        description="Danish Tweets annotated for Hate Speech either being Offensive or not",
        reference="https://aclanthology.org/2020.lrec-1.430/",
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["da"],
        main_score="accuracy",
        revision="59d12749a3c91a186063c7d729ec392fda94681c",
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

    def load_data(self, **kwargs):
        """
        Load dataset from HuggingFace hub
        """
        if self.data_loaded:
            return

        self.dataset = datasets.load_dataset(
            self.metadata_dict["hf_hub_name"],
            revision=self.metadata_dict.get("revision", None),
        )
        self.dataset_transform()
        self.data_loaded = True

    def dataset_transform(self):
        # convert label to a 0/1 label
        labels = self.dataset["train"]["label"]  # type: ignore
        lab2idx = {lab: idx for idx, lab in enumerate(set(labels))}
        self.dataset = self.dataset.map(
            lambda x: {"label": lab2idx[x["label"]]}, remove_columns=["label"]
        )
