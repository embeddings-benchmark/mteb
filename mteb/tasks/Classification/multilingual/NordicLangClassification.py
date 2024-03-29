from __future__ import annotations

import datasets

from mteb.abstasks import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class NordicLangClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="NordicLangClassification",
        description="A dataset for Nordic language identification.",
        reference="https://aclanthology.org/2021.vardial-1.8/",
        dataset={
            "path": "strombergnlp/nordic_langid",
            "revision": "e254179d18ab0165fdb6dbef91178266222bee2a",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["no", "nn"],
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
        n_samples={"test": 3000},
        avg_character_length={"test": 78.2},
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
            name="10k",
            **self.metadata_dict["dataset"],
        )
        self.dataset_transform()
        self.data_loaded = True

    def dataset_transform(self):
        self.dataset = self.dataset.rename_column("sentence", "text")
        self.dataset = self.dataset.rename_column("language", "label")
