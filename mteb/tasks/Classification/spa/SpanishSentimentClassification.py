from __future__ import annotations

from mteb.abstasks import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class SpanishSentimentClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SpanishSentimentClassification",
        description="A Spanish dataset for sentiment classification.",
        reference="https://huggingface.co/datasets/sepidmnorozy/Spanish_sentiment",
        dataset={
            "path": "sepidmnorozy/Spanish_sentiment",
            "revision": "2a6e340e4b59b7c0a78c03a0b79ac27e1b4a2662",
        },
        type="Classification",
        category="s2s",
        modalities=["text"],
        date=("2022-08-16", "2022-08-16"),
        eval_splits=["validation", "test"],
        eval_langs=["spa-Latn"],
        main_score="accuracy",
        domains=["Reviews", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="Not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
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
        descriptive_stats={
            "n_samples": {"validation": 147, "test": 296},
            "avg_character_length": {"validation": 85.02, "test": 87.91},
        },
    )
