from __future__ import annotations

from mteb.abstasks import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class GreekSentimentClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="GreekSentimentClassification",
        dataset={
            "path": "sepidmnorozy/Greek_sentiment",
            "revision": "94b245f3ccdf8e8b2cbf8f13f55eee820b70eccf",
        },
        description="Greek sentiment analysis dataset.",
        reference="https://huggingface.co/datasets/sepidmnorozy/Greek_sentiment",
        type="Classification",
        category="s2s",
        eval_splits=["validation", "test"],
        eval_langs=["ell-Grek"],
        main_score="accuracy",
        date=("2022-08-01", "2022-08-01"),
        form=["written"],
        domains=["Reviews"],
        task_subtypes=["Sentiment/Hate speech"],
        license="Not specified",
        socioeconomic_status="mixed",
        annotations_creators="derived",
        dialect=[],
        text_creation="found",
        bibtex_citation="""
        @inproceedings{mollanorozy-etal-2023-cross,
            title = "Cross-lingual Transfer Learning with \{P\}ersian",
            author = "Mollanorozy, Sepideh  and
            Tanti, Marc  and
            Nissim, Malvina",
            editor = "Beinborn, Lisa  and
            Goswami, Koustava  and
            Murado{\\u{g}}lu, Saliha  and
            Sorokin, Alexey  and
            Kumar, Ritesh  and
            Shcherbakov, Andreas  and
            Ponti, Edoardo M.  and
            Cotterell, Ryan  and
            Vylomova, Ekaterina",
            booktitle = "Proceedings of the 5th Workshop on Research in Computational Linguistic Typology and Multilingual NLP",
            month = may,
            year = "2023",
            address = "Dubrovnik, Croatia",
            publisher = "Association for Computational Linguistics",
            url = "https://aclanthology.org/2023.sigtyp-1.9",
            doi = "10.18653/v1/2023.sigtyp-1.9",
            pages = "89--95",
        }
        """,
        n_samples={"validation": 383, "test": 767},
        avg_character_length={"validation": 208, "test": 206},
    )
