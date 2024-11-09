from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class AfriSentiLangClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="AfriSentiLangClassification",
        description="AfriSentiLID is the largest LID classification dataset for African Languages.",
        dataset={
            "path": "HausaNLP/afrisenti-lid-data",
            "revision": "f17cb5f3ec522ac604601fd09db9fd644ac66ca5",
        },
        reference="https://huggingface.co/datasets/HausaNLP/afrisenti-lid-data/",
        type="Classification",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=[
            "amh-Ethi",  # Amharic (Ethiopic script)
            "arq-Arab",
            "ary-Arab",  # Moroccan Arabic, Standard Arabic (Arabic script)
            "hau-Latn",  # Hausa (Latin script), additional script if written in Ajami (Arabic script)
            "ibo-Latn",  # Igbo (Latin script)
            "kin-Latn",  # Kinyarwanda (Latin script)
            "por-Latn",  # Portuguese (Latin script)
            "pcm-Latn",  # Nigerian Pidgin (Latin script)
            "swa-Latn",  # Swahili (macrolanguage) (Latin script)
            "twi-Latn",  # Twi (Latin script)
            "tso-Latn",  # Tsonga (Latin script)
            "yor-Latn",  # Yoruba (Latin script)
        ],
        main_score="accuracy",
        date=("2023-07-04", "2023-08-04"),
        domains=["Social", "Written"],
        task_subtypes=["Language identification"],
        license="cc-by-nc-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""
        """,
    )

    samples_per_label = 32

    def dataset_transform(self):
        self.dataset = self.dataset.rename_column("tweet", "text")
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["test"]
        )
