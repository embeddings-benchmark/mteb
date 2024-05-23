from __future__ import annotations

import datasets
from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import AbsTaskBitextMining, CrosslingualTask


class DiaBLaBitextMining(AbsTaskBitextMining, CrosslingualTask):
    metadata = TaskMetadata(
        name="DiaBlaBitextMining",
        dataset={
            "path": "rbawden/DiaBLa",
            "revision": "5345895c56a601afe1a98519ce3199be60a27dba",
        },
        description="English-French Parallel Corpus. DiaBLa is an English-French dataset for the evaluation of Machine Translation (MT) for informal, written bilingual dialogue.",
        reference="https://inria.hal.science/hal-03021633",
        type="BitextMining",
        category="s2s",
        eval_splits=["test"],
        eval_langs={
            "fr-en": ["fra-Latn", "eng-Latn"],
            "en-fr": ["eng-Latn", "fra-Latn"],
        },
        main_score="f1",
        date=("2016-01-01", "2017-12-31"),
        form=["written"],
        domains=["Social"],
        task_subtypes=[],
        license="CC BY-NC-SA 4.0",
        socioeconomic_status="mixed",
        annotations_creators="human-annotated",
        dialect=[],
        text_creation="created",
        bibtex_citation="""
        @inproceedings{gonzalez2019diabla,
        title={DiaBLa: A Corpus of Bilingual Spontaneous Written Dialogues for Machine Translation},
        author={González, Matilde and García, Clara and Sánchez, Lucía},
        booktitle={Proceedings of the 12th Language Resources and Evaluation Conference},
        pages={4192--4198},
        year={2019}
        }
        """,
        n_samples={},
        avg_character_length={},
    )

    def load_data(self, **kwargs):
        """Load dataset from HuggingFace hub and convert it to the standard format."""
        if self.data_loaded:
            return

        self.dataset = {}

        for lang in self.hf_subsets:
            self.dataset[lang] = datasets.load_dataset(**self.metadata_dict["dataset"])

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
        for lang in self.hf_subsets:
            self.dataset[lang] = self.dataset[lang].map(create_columns)
