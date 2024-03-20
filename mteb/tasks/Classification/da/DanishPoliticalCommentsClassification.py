import datasets

from mteb.abstasks import AbsTaskClassification


class DanishPoliticalCommentsClassification(AbsTaskClassification):
    metadata = 

@property
    def metadata_dict(self) -> dict[str, str]:
        return dict(self.metadata)
        return {
            "name": "DanishPoliticalCommentsClassification",
            "hf_hub_name": "danish_political_comments",
            "description": "A dataset of Danish political comments rated for sentiment",
            "reference": "NA",
            "type": "Classification",
            "category": "s2s",
            "eval_splits": ["train"],
            "eval_langs": ["da"],
            "main_score": "accuracy",
            "n_experiments": 10,
            "samples_per_label": 16,
            "revision": "edbb03726c04a0efab14fc8c3b8b79e4d420e5a1",
        }

    def load_data(self, **kwargs):
        """
        Load dataset from HuggingFace hub
        """
        if self.data_loaded:
            return

        self.dataset = datasets.load_dataset(
            self.metadata_dict["hf_hub_name"],
            revision=self.metadata_dict.get("revision"),
        )
        self.dataset_transform()
        self.data_loaded = True

    def dataset_transform(self):
        self.dataset = self.dataset.rename_column("sentence", "text")
        self.dataset = self.dataset.rename_column("target", "label")

        # create train and test splits
        self.dataset = self.dataset["train"].train_test_split(0.2, seed=self.seed)
