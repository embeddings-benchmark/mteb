from mteb.abstasks.task_metadata import TaskMetadata
from mteb.abstasks.text.bitext_mining import AbsTaskBitextMining


class DanishMedicinesAgencyBitextMining(AbsTaskBitextMining):
    metadata = TaskMetadata(
        name="DanishMedicinesAgencyBitextMining",
        dataset={
            "path": "mteb/english-danish-parallel-corpus",
            "revision": "0e45e58e9c360134cdb2bf023ad2606a27a2a086",
        },
        description="A Bilingual English-Danish parallel corpus from The Danish Medicines Agency.",
        reference="https://sprogteknologi.dk/dataset/bilingual-english-danish-parallel-corpus-from-the-danish-medicines-agency",
        type="BitextMining",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["dan-Latn", "eng-Latn"],
        main_score="f1",
        date=("2016-01-01", "2019-02-24"),
        domains=["Medical", "Written"],
        task_subtypes=[],
        license="https://opendefinition.org/od/2.1/en/",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation="",
    )
