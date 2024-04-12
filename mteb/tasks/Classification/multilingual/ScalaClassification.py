from __future__ import annotations

from mteb.abstasks import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class ScalaDaClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="ScalaDaClassification",
        description="A modified version of DDT modified for linguistic acceptability classification",
        reference="https://aclanthology.org/2023.nodalida-1.20/",
        dataset={
            "path": "mteb/scala_da_classification",
            "revision": "e60a77795ed5488fb7a03751cf6f2b026fa27a71",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["dan-Latn"],
        main_score="accuracy",
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
        n_samples={"test": 1024},
        avg_character_length={"test": 109.4},
    )

    @property
    def metadata_dict(self) -> dict[str, str]:
        metadata_dict = super().metadata_dict
        metadata_dict["n_experiments"] = 10
        metadata_dict["samples_per_label"] = 32
        return metadata_dict

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
        dataset={
            "path": "mteb/scala_nb_classification",
            "revision": "dda7af4696bd8d5150441908ea8ed6e68a357c13",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["nob-Latn"],
        main_score="accuracy",
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
        n_samples={"test": 1024},
        avg_character_length={"test": 98.4},
    )

    @property
    def metadata_dict(self) -> dict[str, str]:
        metadata_dict = super().metadata_dict
        metadata_dict["n_experiments"] = 10
        metadata_dict["samples_per_label"] = 32
        return metadata_dict

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
        dataset={
            "path": "mteb/scala_nn_classification",
            "revision": "d81637ad324afb995ae395a87055380e8118a9c0",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["nno-Latn"],
        main_score="accuracy",
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
        n_samples={"test": 1024},
        avg_character_length={"test": 104.8},
    )

    @property
    def metadata_dict(self) -> dict[str, str]:
        metadata_dict = super().metadata_dict
        metadata_dict["n_experiments"] = 10
        metadata_dict["samples_per_label"] = 32
        return metadata_dict

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
        dataset={
            "path": "mteb/scala_sv_classification",
            "revision": "aded78beae37445bf3917102d0b332049cfc8c99",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["swe-Latn"],
        main_score="accuracy",
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
        n_samples={"test": 1024},
        avg_character_length={"test": 98.3},
    )

    @property
    def metadata_dict(self) -> dict[str, str]:
        metadata_dict = super().metadata_dict
        metadata_dict["n_experiments"] = 10
        metadata_dict["samples_per_label"] = 32
        return metadata_dict

    def dataset_transform(self):
        # convert label to a 0/1 label
        labels = self.dataset["train"]["label"]  # type: ignore
        lab2idx = {lab: idx for idx, lab in enumerate(set(labels))}
        self.dataset = self.dataset.map(
            lambda x: {"label": lab2idx[x["label"]]}, remove_columns=["label"]
        )
