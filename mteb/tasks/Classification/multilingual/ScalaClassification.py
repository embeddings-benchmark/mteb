from __future__ import annotations

import datasets

from mteb.abstasks import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class ScalaDaClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="ScalaDaClassification",
        description="A modified version of DDT modified for linguistic acceptability classification",
        reference="https://aclanthology.org/2023.nodalida-1.20/",
        dataset={
            "path": "ScandEval/scala-da",
            "revision": "1de08520a7b361e92ffa2a2201ebd41942c54675",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["da"],
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
        metadata_dict = dict(self.metadata)
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
            "path": "ScandEval/scala-nb",
            "revision": "237111a078ad5a834a55c57803d40bbe410ed03b",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["nb"],
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
        metadata_dict = dict(self.metadata)
        metadata_dict["n_experiments"] = 10
        metadata_dict["samples_per_label"] = 32
        return dict(self.metadata)

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
            "path": "ScandEval/scala-nn",
            "revision": "9d9a2a4092ed3cacf0744592f6d2f32ab8ef4c0b",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["nn"],
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
        metadata_dict = dict(self.metadata)
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
            "path": "ScandEval/scala-sv",
            "revision": "1b48e3dcb02872335ff985ff938a054a4ed99008",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["sv"],
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
        metadata_dict = dict(self.metadata)
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
