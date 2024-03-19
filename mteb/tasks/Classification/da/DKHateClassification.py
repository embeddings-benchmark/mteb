import datasets

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification


class DKHateClassification(AbsTaskClassification):
    @property
    def metadata_dict(self):
        return {
            "name": "DKHateClassification",
            "hf_hub_name": "DDSC/dkhate",
            "description": "Danish Tweets annotated for Hate Speech either being Offensive or not",
            "reference": "https://aclanthology.org/2020.lrec-1.430/",
            "type": "Classification",
            "category": "s2s",
            "eval_splits": ["test"],
            "eval_langs": ["da"],
            "main_score": "accuracy",
            "n_experiments": 10,
            "samples_per_label": 16,
            "revision": "59d12749a3c91a186063c7d729ec392fda94681c",
        }

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
