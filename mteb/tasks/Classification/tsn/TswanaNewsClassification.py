from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class TswanaNewsClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="TswanaNewsClassification",
        description="Tswana News Classification Dataset",
        reference="https://link.springer.com/chapter/10.1007/978-3-031-49002-6_17",
        dataset={
            "path": "dsfsi/daily-news-dikgang",
            "revision": "061ca1525717eebaaa9bada240f6cbb31eb3aa87",
        },
        type="Classification",
        task_subtypes=["Topic classification"],
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["tsn-Latn"],
        main_score="accuracy",
        date=("2015-01-01", "2023-01-01"),
        domains=["News", "Written"],
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""
        @inproceedings{marivate2023puoberta,
            title   = {PuoBERTa: Training and evaluation of a curated language model for Setswana},
            author  = {Vukosi Marivate and Moseli Mots'Oehli and Valencia Wagner and Richard Lastrucci and Isheanesu Dzingirai},
            year    = {2023},
            booktitle= {SACAIR 2023 (To Appear)},
            keywords = {NLP},
            preprint_url = {https://arxiv.org/abs/2310.09141},
            dataset_url = {https://github.com/dsfsi/PuoBERTa},
            software_url = {https://huggingface.co/dsfsi/PuoBERTa}
        }
        """,
    )
