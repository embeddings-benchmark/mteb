import datasets

from mteb.abstasks import AbsTaskBitextMining


class NorwegianCourtsBitextMining(AbsTaskBitextMining):
    @property
    def metadata_dict(self):
        return {
            "name": "NorwegianCourtsBitextMining",
            "hf_hub_name": "kardosdrur/norwegian-courts",
            "description": "Nynorsk and Bokmål parallel corpus from Norwegian courts. "
            + "Norway has two standardised written languages. "
            + "Bokmål is a variant closer to Danish, while Nynorsk was created to resemble "
            + "regional dialects of Norwegian.",
            "reference": "https://opus.nlpl.eu/index.php",
            "type": "BitextMining",
            "category": "s2s",
            "eval_splits": ["test"],
            "eval_langs": ["nb", "nn"],
            "main_score": "f1",
            "revision": "d79af07e969a6678fcbbe819956840425816468f",
        }

    def load_data(self, **kwargs):
        """
        Load dataset from HuggingFace hub and convert it to the standard format.
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
        # Convert to standard format
        self.dataset = self.dataset.rename_column("nb", "sentence1")
        self.dataset = self.dataset.rename_column("nn", "sentence2")
