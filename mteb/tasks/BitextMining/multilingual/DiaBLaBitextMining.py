from __future__ import annotations

import datasets

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import AbsTaskBitextMining, CrosslingualTask


class DiaBLaBitextMining(AbsTaskBitextMining, CrosslingualTask):
    metadata = TaskMetadata(
        name="DiaBlaBitextMining",
        hf_hub_name="rbawden/DiaBLa",
        description="English-French Parallel Corpus. DiaBLa is an English-French dataset for the evaluation of Machine Translation (MT) for informal, written bilingual dialogue.",
        reference="https://inria.hal.science/hal-03021633",
        type="BitextMining",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["fr-en", "en-fr"],
        main_score="f1",
        revision="5345895c56a601afe1a98519ce3199be60a27dba",
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
        n_samples=None,
        avg_character_length=None,
    )

    def load_data(self, **kwargs):
        """
        Load dataset from HuggingFace hub and convert it to the standard format.
        """
        if self.data_loaded:
            return

        self.dataset = {}
        for lang in self.langs:
            self.dataset[lang] = datasets.load_dataset(
                self.metadata_dict["hf_hub_name"],
                revision=self.metadata_dict.get("revision", None),
            )

        self.dataset_transform()
        self.data_loaded = True

    def dataset_transform(self):
        def create_columns(row):
            """Put all French texts in column 'sentence1' and English texts in 'sentence2' column"""
            row["orig_lang"] = row["utterance_meta"]["lang"]
            row["sentence1"] = (
                row["orig"] if row["orig_lang"] == "french" else row["ref"]
            )
            row["sentence2"] = (
                row["orig"] if not row["orig_lang"] == "french" else row["ref"]
            )
            return row

        # Convert to standard format
        for lang in self.langs:
            self.dataset[lang] = self.dataset[lang].map(create_columns)
