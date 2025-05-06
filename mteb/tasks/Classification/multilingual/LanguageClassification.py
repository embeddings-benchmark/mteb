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
        bibtex_citation=r"""
@inproceedings{conneau2018xnli,
  author = {Conneau, Alexis
and Rinott, Ruty
and Lample, Guillaume
and Williams, Adina
and Bowman, Samuel R.
and Schwenk, Holger
and Stoyanov, Veselin},
  booktitle = {Proceedings of the 2018 Conference on Empirical Methods
in Natural Language Processing},
  location = {Brussels, Belgium},
  publisher = {Association for Computational Linguistics},
  title = {XNLI: Evaluating Cross-lingual Sentence Representations},
  year = {2018},
}
""",
    )

    def dataset_transform(self) -> None:
        self.dataset = self.dataset.rename_columns({"labels": "label"})
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["test"]
        )
