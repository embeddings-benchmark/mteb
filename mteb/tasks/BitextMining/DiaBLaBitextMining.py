import json
import datasets

from ...abstasks import AbsTaskBitextMining, CrosslingualTask


class DiaBLaBitextMining(AbsTaskBitextMining, CrosslingualTask):
    @property
    def description(self):
        return {
            "name": "DiaBLaBitextMining",
            "hf_hub_name": "rbawden/DiaBLa",
            "description": (
                "English-French Parallel Corpus. "
                + "DiaBLa is an English-French dataset for the evaluation of Machine Translation (MT) for informal,"
                " written bilingual dialogue."
            ),
            "reference": "https://inria.hal.science/hal-03021633",
            "type": "BitextMining",
            "category": "s2s",
            "eval_splits": ["test"],
            "eval_langs": ["fr", "en"],
            "main_score": "f1",
            "revision": "5345895c56a601afe1a98519ce3199be60a27dba",
        }

    def load_data(self, **kwargs):
        """
        Load dataset from HuggingFace hub and convert it to the standard format.
        """
        if self.data_loaded:
            return

        self.dataset = {}
        for lang in self.langs:
            print(lang)
            self.dataset[lang] = datasets.load_dataset(
                self.description["hf_hub_name"],
                revision=self.description.get("revision", None),
            )

        self.dataset_transform()
        self.data_loaded = True

    def dataset_transform(self):
        def create_columns(row):
            """Put all French texts in column 'sentence1' and English texts in 'sentence2' column"""
            row["orig_lang"] = row["utterance_meta"]["lang"]
            row["sentence1"] = row["orig"] if row["orig_lang"] == "french" else row["ref"]
            row["sentence2"] = row["orig"] if not row["orig_lang"] == "french" else row["ref"]
            return row

        # Convert to standard format
        for lang in self.langs:
            self.dataset[lang] = self.dataset[lang].map(create_columns)
            if lang == "en":
                self.dataset[lang]["test"] = self.dataset[lang]["test"].rename_column("sentence2", "sentence_1")
                self.dataset[lang]["test"] = self.dataset[lang]["test"].rename_column("sentence1", "sentence2")
                self.dataset[lang]["test"] = self.dataset[lang]["test"].rename_column("sentence_1", "sentence1")
