from __future__ import annotations

from mteb.abstasks import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata

N_SAMPLES = 2800


class SlovakSentimentClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SlovakSentimentClassification",
        description="Slovak Sentiment Classification Dataset",
        reference="https://huggingface.co/datasets/sepidmnorozy/Slovak_sentiment",
        dataset={
            "path": "sepidmnorozy/Slovak_sentiment",
            "revision": "e698d1df52766d73ae1cc569dfc622527c329a08",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["slk-Latn"],
        main_score="accuracy",
        date=("2022-08-01", "2022-08-01"),
        form=["written"],
        domains=["Reviews"],
        task_subtypes=["Sentiment/Hate speech"],
        license="Not specified",
        socioeconomic_status="medium",
        annotations_creators="human-annotated",
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
        n_samples={"validation": 522, "test": 1040},
        avg_character_length={"validation": 84.96, "test": 91.95},
    )
