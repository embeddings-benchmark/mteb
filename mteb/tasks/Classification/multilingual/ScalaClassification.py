import datasets

from mteb.abstasks import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class ScalaDaClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="ScalaDaClassification",
        description="A modified version of DDT modified for linguistic acceptability classification",
        reference="https://aclanthology.org/2023.nodalida-1.20/",
        hf_hub_name="ScandEval/scala-da",
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["da"],
        main_score="accuracy",
        revision="1de08520a7b361e92ffa2a2201ebd41942c54675",
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
        metadata_dict["samples_per_label"] = 32
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


class ScalaNbClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="ScalaNbClassification",
        description="A Norwegian dataset for linguistic acceptability classification for Bokmål",
        reference="https://aclanthology.org/2023.nodalida-1.20/",
        hf_hub_name="ScandEval/scala-nb",
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["nb"],
        main_score="accuracy",
        revision="237111a078ad5a834a55c57803d40bbe410ed03b",
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
        metadata_dict["samples_per_label"] = 32
        return dict(self.metadata)

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


class ScalaNnClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="ScalaNnClassification",
        description="A Norwegian dataset for linguistic acceptability classification for Nynorsk",
        reference="https://aclanthology.org/2023.nodalida-1.20/",
        hf_hub_name="ScandEval/scala-nn",
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["nn"],
        main_score="accuracy",
        revision="9d9a2a4092ed3cacf0744592f6d2f32ab8ef4c0b",
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
        metadata_dict["samples_per_label"] = 32
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


class ScalaSvClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="ScalaSvClassification",
        description="A Swedish dataset for linguistic acceptability classification",
        reference="https://aclanthology.org/2023.nodalida-1.20/",
        hf_hub_name="ScandEval/scala-sv",
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["sv"],
        main_score="accuracy",
        revision="1b48e3dcb02872335ff985ff938a054a4ed99008",
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
        metadata_dict["samples_per_label"] = 32
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
