from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import AbsTaskSTS, MultilingualTask

_LANGUAGES = {
    "ko-ko": ["kor-Hang"],
    "ar-ar": ["ara-Arab"],
    "en-ar": ["eng-Latn", "ara-Arab"],
    "en-de": ["eng-Latn", "deu-Latn"],
    "en-en": ["eng-Latn"],
    "en-tr": ["eng-Latn", "tur-Latn"],
    "es-en": ["spa-Latn", "eng-Latn"],
    "es-es": ["spa-Latn"],
    "fr-en": ["fra-Latn", "eng-Latn"],
    "it-en": ["ita-Latn", "eng-Latn"],
    "nl-en": ["nld-Latn", "eng-Latn"],
}


class STS17Crosslingual(AbsTaskSTS, MultilingualTask):
    fast_loading = True
    metadata = TaskMetadata(
        name="STS17",
        dataset={
            "path": "mteb/sts17-crosslingual-sts",
            "revision": "faeb762787bd10488a50c8b5be4a3b82e411949c",
        },
        description="Semeval-2017 task 1: Semantic textual similarity-multilingual and cross-lingual focused evaluation",
        reference="https://alt.qcri.org/semeval2017/task1/",
        type="STS",
        category="s2s",
        eval_splits=["test"],
        eval_langs=_LANGUAGES,
        main_score="cosine_spearman",
        date=("2014-01-01", "2015-12-31"),
        form=["written"],
        domains=["News", "Web"],
        task_subtypes=[],
        license="Not specified",
        socioeconomic_status="mixed",
        annotations_creators="human-annotated",
        dialect=[],
        text_creation="created",
        bibtex_citation="""@inproceedings{cer-etal-2017-semeval,
    title = "{S}em{E}val-2017 Task 1: Semantic Textual Similarity Multilingual and Crosslingual Focused Evaluation",
    author = "Cer, Daniel  and
      Diab, Mona  and
      Agirre, Eneko  and
      Lopez-Gazpio, I{\~n}igo  and
      Specia, Lucia",
    editor = "Bethard, Steven  and
      Carpuat, Marine  and
      Apidianaki, Marianna  and
      Mohammad, Saif M.  and
      Cer, Daniel  and
      Jurgens, David",
    booktitle = "Proceedings of the 11th International Workshop on Semantic Evaluation ({S}em{E}val-2017)",
    month = aug,
    year = "2017",
    address = "Vancouver, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/S17-2001",
    doi = "10.18653/v1/S17-2001",
    pages = "1--14",
    abstract = "Semantic Textual Similarity (STS) measures the meaning similarity of sentences. Applications include machine translation (MT), summarization, generation, question answering (QA), short answer grading, semantic search, dialog and conversational systems. The STS shared task is a venue for assessing the current state-of-the-art. The 2017 task focuses on multilingual and cross-lingual pairs with one sub-track exploring MT quality estimation (MTQE) data. The task obtained strong participation from 31 teams, with 17 participating in \textit{all language tracks}. We summarize performance and review a selection of well performing methods. Analysis highlights common errors, providing insight into the limitations of existing models. To support ongoing work on semantic representations, the \textit{STS Benchmark} is introduced as a new shared training and evaluation set carefully selected from the corpus of English STS shared task data (2012-2017).",
}""",
        n_samples={"test": 500},
        avg_character_length={"test": 43.3},
    )

    @property
    def metadata_dict(self) -> dict[str, str]:
        metadata_dict = super().metadata_dict
        metadata_dict["min_score"] = 0
        metadata_dict["max_score"] = 5
        return metadata_dict
