from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.MultilingualTask import MultilingualTask
from mteb.abstasks.TaskMetadata import TaskMetadata

_LANGUAGES = {
    "da": ["dan-Latn"],
    "en": ["eng-Latn"],
    "fi": ["fin-Latn"],
    "no": ["nob-Latn"],
    "sv": ["swe-Latn"],
}


class ScandiSentClassification(MultilingualTask, AbsTaskClassification):
    metadata = TaskMetadata(
        name="ScandiSentClassification",
        dataset={
            "path": "mteb/scandisent",
            "revision": "97672414ad7453a106edfbfb1a0ceb152355b9dd",
        },
        description="The corpus is crawled from se.trustpilot.com, no.trustpilot.com, dk.trustpilot.com, fi.trustpilot.com and trustpilot.com.",
        reference="https://github.com/timpal0l/ScandiSent",
        type="Classification",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=_LANGUAGES,
        main_score="accuracy",
        date=("2020-09-01", "2022-04-09"),
        domains=["Reviews", "Written"],
        dialect=[],
        task_subtypes=["Sentiment/Hate speech"],
        license="openrail",
        annotations_creators="expert-annotated",
        sample_creation="found",
        bibtex_citation="""@inproceedings{isbister-etal-2021-stop,
            title = "Should we Stop Training More Monolingual Models, and Simply Use Machine Translation Instead?",
            author = "Isbister, Tim  and
            Carlsson, Fredrik  and
            Sahlgren, Magnus",
            editor = "Dobnik, Simon  and
            {\O}vrelid, Lilja",
            booktitle = "Proceedings of the 23rd Nordic Conference on Computational Linguistics (NoDaLiDa)",
            month = may # " 31--2 " # jun,
            year = "2021",
            address = "Reykjavik, Iceland (Online)",
            publisher = {Link{\"o}ping University Electronic Press, Sweden},
            url = "https://aclanthology.org/2021.nodalida-main.42/",
            pages = "385--390",
        }""",
    )
