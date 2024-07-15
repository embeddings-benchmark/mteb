from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import AbsTaskClassification

_LANGUAGES = {
    "ara": ["ara-Arab"],
    "bul": ["bul-Cyrl"],
    "deu": ["deu-Latn"],
    "ell": ["ell-Grek"],
    "eng": ["eng-Latn"],
    "spa": ["spa-Latn"],
    "fra": ["fra-Latn"],
    "hin": ["hin-Deva"],
    "ita": ["ita-Latn"],
    "jpn": ["jpn-Jpan"],
    "nld": ["nld-Latn"],
    "pol": ["pol-Latn"],
    "por": ["por-Latn"],
    "rus": ["rus-Cyrl"],
    "swa": ["swa-Latn"],
    "tha": ["tha-Thai"],
    "tur": ["tur-Latn"],
    "urd": ["urd-Arab"],
    "vie": ["vie-Latn"],
    "zho": ["cmn-Hans"],
}


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
            "avg_character_length": {"test": 107.8},
        },
    )

    def dataset_transform(self) -> None:
        self.dataset = self.dataset.rename_columns({"labels": "label"})
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["test"]
        )
