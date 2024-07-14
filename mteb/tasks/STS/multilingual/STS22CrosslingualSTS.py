from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import AbsTaskSTS, MultilingualTask

_LANGUAGES = {
    "en": ["eng-Latn"],
    "de": ["deu-Latn"],
    "es": ["spa-Latn"],
    "pl": ["pol-Latn"],
    "tr": ["tur-Latn"],
    "ar": ["ara-Arab"],
    "ru": ["rus-Cyrl"],
    "zh": ["cmn-Hans"],
    "fr": ["fra-Latn"],
    "de-en": ["deu-Latn", "eng-Latn"],
    "es-en": ["spa-Latn", "eng-Latn"],
    "it": ["ita-Latn"],
    "pl-en": ["pol-Latn", "eng-Latn"],
    "zh-en": ["cmn-Hans", "eng-Latn"],
    "es-it": ["spa-Latn", "ita-Latn"],
    "de-fr": ["deu-Latn", "fra-Latn"],
    "de-pl": ["deu-Latn", "pol-Latn"],
    "fr-pl": ["fra-Latn", "pol-Latn"],
}


class STS22CrosslingualSTSv2(AbsTaskSTS, MultilingualTask):
    fast_loading = True
    metadata = TaskMetadata(
        name="STS22.v2",
        dataset={
            "path": "mteb/sts22-crosslingual-sts",
            "revision": "d31f33a128469b20e357535c39b82fb3c3f6f2bd",
        },
        description="SemEval 2022 Task 8: Multilingual News Article Similarity. Version 2 filters updated on STS22 by removing pairs where one of entries contain empty sentences.",
        reference="https://competitions.codalab.org/competitions/33835",
        type="STS",
        category="p2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=_LANGUAGES,
        main_score="cosine_spearman",
        date=("2020-01-01", "2020-06-11"),
        domains=["News", "Written"],
        task_subtypes=[],
        license="Not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@inproceedings{chen-etal-2022-semeval,
    title = "{S}em{E}val-2022 Task 8: Multilingual news article similarity",
    author = {Chen, Xi  and
      Zeynali, Ali  and
      Camargo, Chico  and
      Fl{\"o}ck, Fabian  and
      Gaffney, Devin  and
      Grabowicz, Przemyslaw  and
      Hale, Scott  and
      Jurgens, David  and
      Samory, Mattia},
    editor = "Emerson, Guy  and
      Schluter, Natalie  and
      Stanovsky, Gabriel  and
      Kumar, Ritesh  and
      Palmer, Alexis  and
      Schneider, Nathan  and
      Singh, Siddharth  and
      Ratan, Shyam",
    booktitle = "Proceedings of the 16th International Workshop on Semantic Evaluation (SemEval-2022)",
    month = jul,
    year = "2022",
    address = "Seattle, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.semeval-1.155",
    doi = "10.18653/v1/2022.semeval-1.155",
    pages = "1094--1106",
}""",
        descriptive_stats={
            "n_samples": {"test": 3958},
            "avg_character_length": {"test": 1993.6},
        },
    )

    @property
    def metadata_dict(self) -> dict[str, str]:
        metadata_dict = super().metadata_dict
        metadata_dict["min_score"] = 1
        metadata_dict["max_score"] = 4
        return metadata_dict


class STS22CrosslingualSTS(AbsTaskSTS, MultilingualTask):
    superseded_by = "STS22.v2"
    fast_loading = True
    metadata = TaskMetadata(
        name="STS22",
        dataset={
            "path": "mteb/sts22-crosslingual-sts",
            "revision": "de9d86b3b84231dc21f76c7b7af1f28e2f57f6e3",
        },
        description="SemEval 2022 Task 8: Multilingual News Article Similarity",
        reference="https://competitions.codalab.org/competitions/33835",
        type="STS",
        category="p2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=_LANGUAGES,
        main_score="cosine_spearman",
        date=("2020-01-01", "2020-06-11"),
        domains=["News", "Written"],
        task_subtypes=[],
        license="Not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@inproceedings{chen-etal-2022-semeval,
    title = "{S}em{E}val-2022 Task 8: Multilingual news article similarity",
    author = {Chen, Xi  and
      Zeynali, Ali  and
      Camargo, Chico  and
      Fl{\"o}ck, Fabian  and
      Gaffney, Devin  and
      Grabowicz, Przemyslaw  and
      Hale, Scott  and
      Jurgens, David  and
      Samory, Mattia},
    editor = "Emerson, Guy  and
      Schluter, Natalie  and
      Stanovsky, Gabriel  and
      Kumar, Ritesh  and
      Palmer, Alexis  and
      Schneider, Nathan  and
      Singh, Siddharth  and
      Ratan, Shyam",
    booktitle = "Proceedings of the 16th International Workshop on Semantic Evaluation (SemEval-2022)",
    month = jul,
    year = "2022",
    address = "Seattle, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.semeval-1.155",
    doi = "10.18653/v1/2022.semeval-1.155",
    pages = "1094--1106",
}""",
        descriptive_stats={
            "n_samples": {"test": 8056},
            "avg_character_length": {"test": 1993.6},
        },
    )

    @property
    def metadata_dict(self) -> dict[str, str]:
        metadata_dict = super().metadata_dict
        metadata_dict["min_score"] = 1
        metadata_dict["max_score"] = 4
        return metadata_dict
