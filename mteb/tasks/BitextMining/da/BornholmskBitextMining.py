import datasets

from mteb.abstasks import AbsTaskBitextMining


class BornholmBitextMining(AbsTaskBitextMining):
    @property
    def description(self):
        return {
            "name": "BornholmBitextMining",
            "hf_hub_name": "strombergnlp/bornholmsk_parallel",
            "description": "Danish Bornholmsk Parallel Corpus. "
            + "Bornholmsk is a Danish dialect spoken on the island of Bornholm, Denmark. "
            + "Historically it is a part of east Danish which was also spoken in Scania and Halland, Sweden.",
            "reference": "https://aclanthology.org/W19-6138/",
            "type": "BitextMining",
            "category": "s2s",
            "eval_splits": ["test"],
            "eval_langs": ["da", "da-bornholm"],
            "main_score": "f1",
            "revision": "3bc5cfb4ec514264fe2db5615fac9016f7251552",
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
        self.dataset = self.dataset.rename_column("da", "sentence1")
        self.dataset = self.dataset.rename_column("da_bornholm", "sentence2")


if __name__ == "__main__":
    dataset = BornholmBitextMining()
    dataset.load_data()

    pass
