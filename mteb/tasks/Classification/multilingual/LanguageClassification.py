from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import AbsTaskClassification

_LANGUAGES = [
    "ara-Arab",
    "bul-Cyrl",
    "deu-Latn",
    "ell-Grek",
    "eng-Latn",
    "spa-Latn",
    "fra-Latn",
    "hin-Deva",
    "ita-Latn",
    "jpn-Jpan",
    "nld-Latn",
    "pol-Latn",
    "por-Latn",
    "rus-Cyrl",
    "swa-Latn",
    "tha-Thai",
    "tur-Latn",
    "urd-Arab",
    "vie-Latn",
    "cmn-Hans",
]


class LanguageClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="LanguageClassification",
        dataset={
            "path": "papluca/language-identification",
            "revision": "aa56583bf2bc52b0565770607d6fc3faebecf9e2",
        },
        description="A language identification dataset for 20 languages.",
        reference="https://huggingface.co/datasets/papluca/language-identification",
        category="s2s",
        modalities=["text"],
        type="Classification",
        eval_splits=["test"],
        eval_langs=_LANGUAGES,
        main_score="accuracy",
        date=("2021-11-01", "2021-11-30"),
        domains=["Reviews", "Web", "Non-fiction", "Fiction", "Government", "Written"],
        task_subtypes=["Language identification"],
        license="Not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@InProceedings{conneau2018xnli,
  author = {Conneau, Alexis
                 and Rinott, Ruty
                 and Lample, Guillaume
                 and Williams, Adina
                 and Bowman, Samuel R.
                 and Schwenk, Holger
                 and Stoyanov, Veselin},
  title = {XNLI: Evaluating Cross-lingual Sentence Representations},
  booktitle = {Proceedings of the 2018 Conference on Empirical Methods
               in Natural Language Processing},
  year = {2018},
  publisher = {Association for Computational Linguistics},
  location = {Brussels, Belgium},
}""",
        descriptive_stats={
            "n_samples": {"test": 2048},
            "avg_character_length": {
                "test": {
                    "num_texts": 2048,
                    "num_labels": 2048,
                    "average_text_length": 109.546875,
                    "num_label_17": 102,
                    "num_label_0": 102,
                    "num_label_11": 102,
                    "num_label_4": 103,
                    "num_label_3": 102,
                    "num_label_1": 102,
                    "num_label_10": 102,
                    "num_label_2": 103,
                    "num_label_16": 103,
                    "num_label_9": 103,
                    "num_label_5": 102,
                    "num_label_7": 102,
                    "num_label_13": 102,
                    "num_label_14": 103,
                    "num_label_12": 102,
                    "num_label_15": 103,
                    "num_label_19": 102,
                    "num_label_18": 102,
                    "num_label_6": 103,
                    "num_label_8": 103,
                },
                "train": {
                    "num_texts": 70000,
                    "num_labels": 70000,
                    "average_text_length": 110.86141428571429,
                    "num_label_12": 3500,
                    "num_label_1": 3500,
                    "num_label_19": 3500,
                    "num_label_15": 3500,
                    "num_label_13": 3500,
                    "num_label_11": 3500,
                    "num_label_17": 3500,
                    "num_label_14": 3500,
                    "num_label_16": 3500,
                    "num_label_5": 3500,
                    "num_label_0": 3500,
                    "num_label_8": 3500,
                    "num_label_7": 3500,
                    "num_label_2": 3500,
                    "num_label_3": 3500,
                    "num_label_10": 3500,
                    "num_label_6": 3500,
                    "num_label_18": 3500,
                    "num_label_4": 3500,
                    "num_label_9": 3500,
                },
            },
        },
    )

    def dataset_transform(self) -> None:
        self.dataset = self.dataset.rename_columns({"labels": "label"})
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["test"]
        )
