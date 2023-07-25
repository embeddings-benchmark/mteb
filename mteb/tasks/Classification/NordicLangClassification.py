import datasets

from mteb.abstasks import AbsTaskClassification


class NordicLangClassification(AbsTaskClassification):
    @property
    def description(self):
        return {
            "name": "NordicLangClassification",
            "hf_hub_name": "strombergnlp/nordic_langid",
            "description": "A dataset for Nordic language identification.",
            "reference": "https://aclanthology.org/2021.vardial-1.8/",
            "type": "Classification",
            "category": "s2s",
            "eval_splits": ["test"],
            "eval_langs": ["da", "no", "sv", "nb", "no", "is", "fo"],
            "main_score": "accuracy",
            "n_experiments": 10,
            "samples_per_label": 32,
            "revision": "e254179d18ab0165fdb6dbef91178266222bee2a",
        }

    def load_data(self, **kwargs):
        """
        Load dataset from HuggingFace hub
        """
        if self.data_loaded:
            return

        self.dataset = datasets.load_dataset(
            self.description["hf_hub_name"], 
            "10k", # select relevant subset
            revision=self.description.get("revision")  
        )
        self.dataset_transform()
        self.data_loaded = True

    def dataset_transform(self):
        self.dataset = self.dataset.rename_column("sentence", "text")
        self.dataset = self.dataset.rename_column("language", "label")

