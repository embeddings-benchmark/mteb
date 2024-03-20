import datasets

from mteb.abstasks import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class NordicLangClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="NordicLangClassification",
        description="A dataset for Nordic language identification.",
        reference="https://aclanthology.org/2021.vardial-1.8/",
        hf_hub_name="strombergnlp/nordic_langid",
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["no", "nn"],
        main_score="accuracy",
        revision="e254179d18ab0165fdb6dbef91178266222bee2a",
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
            "10k",
            revision=self.metadata_dict.get("revision"),  # select relevant subset
        )
        self.dataset_transform()
        self.data_loaded = True

    def dataset_transform(self):
        self.dataset = self.dataset.rename_column("sentence", "text")
        self.dataset = self.dataset.rename_column("language", "label")
