from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class CyrillicTurkicLangClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="CyrillicTurkicLangClassification",
        description="Cyrillic dataset of 8 Turkic languages spoken in Russia and former USSR",
        dataset={
            "path": "tatiana-merz/cyrillic_turkic_langs",
            "revision": "e42d330f33d65b7b72dfd408883daf1661f06f18",
        },
        reference="https://huggingface.co/datasets/tatiana-merz/cyrillic_turkic_langs",
        type="Classification",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=[
            "bak-Cyrl",  # Bashkir
            "chv-Cyrl",  # Chuvash
            "tat-Cyrl",  # Tatar
            "kir-Cyrl",  # Kyrgyz
            "rus-Cyrl",  # Russian
            "kaz-Cyrl",  # Kazakh
            "tyv-Cyrl",  # Tuvinian
            "krc-Cyrl",  # Karachay-Balkar
            "sah-Cyrl",  # Yakut
        ],
        main_score="accuracy",
        date=("1998-01-01", "2012-05-01"),
        domains=["Web", "Written"],
        task_subtypes=["Language identification"],
        license="cc-by-nc-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""
        @inproceedings{goldhahn2012building,
        title={Building Large Monolingual Dictionaries at the Leipzig Corpora Collection: From 100 to 200 Languages},
        author={Goldhahn, Dirk and Eckart, Thomas and Quasthoff, Uwe},
        booktitle={Proceedings of the Eighth International Conference on Language Resources and Evaluation (LREC'12)},
        year={2012}
        }
        """,
        descriptive_stats={
            "n_samples": {"test": 2048},
            "avg_character_length": {"test": 92.22},
        },
    )

    def dataset_transform(self):
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["test"]
        )
