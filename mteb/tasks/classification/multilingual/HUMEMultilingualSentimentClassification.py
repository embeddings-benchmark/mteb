from mteb.abstasks import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata

_LANGUAGES = {
    "eng": ["eng-Latn"],
    "ara": ["ara-Arab"],
    "nor": ["nob-Latn"],
    "rus": ["rus-Cyrl"],
}


class HUMEMultilingualSentimentClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="HUMEMultilingualSentimentClassification",
        dataset={
            "path": "mteb/HUMEMultilingualSentimentClassification",
            "revision": "1b988d30980efdd9c27de1643837bf3ae5bae814",
        },
        description=(
            "Human evaluation subset of Sentiment classification dataset with binary "
            "(positive vs negative sentiment) labels. Includes 4 languages."
        ),
        reference="https://huggingface.co/datasets/mteb/multilingual-sentiment-classification",
        type="Classification",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=_LANGUAGES,
        main_score="accuracy",
        date=("2022-08-01", "2022-08-01"),
        domains=["Reviews", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="not specified",
        annotations_creators="derived",
        dialect=["ar-dz"],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{mollanorozy-etal-2023-cross,
  address = {Dubrovnik, Croatia},
  author = {Mollanorozy, Sepideh  and
Tanti, Marc  and
Nissim, Malvina},
  booktitle = {Proceedings of the 5th Workshop on Research in Computational Linguistic Typology and Multilingual NLP},
  doi = {10.18653/v1/2023.sigtyp-1.9},
  editor = {Beinborn, Lisa  and
Goswami, Koustava  and
Murado{\\u{g}}lu, Saliha  and
Sorokin, Alexey  and
Kumar, Ritesh  and
Shcherbakov, Andreas  and
Ponti, Edoardo M.  and
Cotterell, Ryan  and
Vylomova, Ekaterina},
  month = may,
  pages = {89--95},
  publisher = {Association for Computational Linguistics},
  title = {Cross-lingual Transfer Learning with \{P\}ersian},
  url = {https://aclanthology.org/2023.sigtyp-1.9},
  year = {2023},
}
""",
        adapted_from=["MultilingualSentimentClassification"],
    )
