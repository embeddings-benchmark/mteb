import datasets

from ...abstasks import AbsTaskBitextMining


class DiaBLaBitextMining(AbsTaskBitextMining):
    @property
    def description(self):
        return {
            "name": "DiaBLaBitextMining",
            "hf_hub_name": "rbawden/DiaBLa",
            "description": (
                "English-French Parallel Corpus. "
                + "DiaBLa is an Englis-French dataset for the evaluation of Machine Translation (MT) for informal,"
                " written bilingual dialogue."
            ),
            "reference": "https://inria.hal.science/hal-03021633",
            "type": "BitextMining",
            "category": "s2s",
            "eval_splits": ["test"],
            "eval_langs": ["fr"],
            "main_score": "f1",
            "revision": "5345895c56a601afe1a98519ce3199be60a27dba",
        }

    def load_data(self, **kwargs):
        """
        Load dataset from HuggingFace hub and convert it to the standard format.
        """
        if self.data_loaded:
            return

        self.dataset = datasets.load_dataset(
            self.description["hf_hub_name"],
            revision=self.description.get("revision", None),
        )
        self.dataset_transform()
        self.data_loaded = True

    def dataset_transform(self):
        # Convert to standard format
        self.dataset = self.dataset.rename_column("orig", "sentence1")
        self.dataset = self.dataset.rename_column("ref", "sentence2")
