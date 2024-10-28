from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata

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
        license="not specified",
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
            "test": {
                "num_samples": 2048,
                "average_text_length": 109.546875,
                "unique_labels": 20,
                "labels": {
                    "17": {"count": 102},
                    "0": {"count": 102},
                    "11": {"count": 102},
                    "4": {"count": 103},
                    "3": {"count": 102},
                    "1": {"count": 102},
                    "10": {"count": 102},
                    "2": {"count": 103},
                    "16": {"count": 103},
                    "9": {"count": 103},
                    "5": {"count": 102},
                    "7": {"count": 102},
                    "13": {"count": 102},
                    "14": {"count": 103},
                    "12": {"count": 102},
                    "15": {"count": 103},
                    "19": {"count": 102},
                    "18": {"count": 102},
                    "6": {"count": 103},
                    "8": {"count": 103},
                },
            },
            "train": {
                "num_samples": 70000,
                "average_text_length": 110.86141428571429,
                "unique_labels": 20,
                "labels": {
                    "12": {"count": 3500},
                    "1": {"count": 3500},
                    "19": {"count": 3500},
                    "15": {"count": 3500},
                    "13": {"count": 3500},
                    "11": {"count": 3500},
                    "17": {"count": 3500},
                    "14": {"count": 3500},
                    "16": {"count": 3500},
                    "5": {"count": 3500},
                    "0": {"count": 3500},
                    "8": {"count": 3500},
                    "7": {"count": 3500},
                    "2": {"count": 3500},
                    "3": {"count": 3500},
                    "10": {"count": 3500},
                    "6": {"count": 3500},
                    "18": {"count": 3500},
                    "4": {"count": 3500},
                    "9": {"count": 3500},
                },
            },
        },
    )

    def dataset_transform(self) -> None:
        self.dataset = self.dataset.rename_columns({"labels": "label"})
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["test"]
        )
