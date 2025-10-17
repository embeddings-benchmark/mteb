from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class JapaneseSentimentClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="JapaneseSentimentClassification",
        dataset={
            "path": "mteb/JapaneseSentimentClassification",
            "revision": "113a72dd629207b956dd4db3c2d11445853f3b1f",
        },
        description=(
            "Japanese sentiment classification dataset with binary\n"
            "(positive vs negative sentiment) labels. This version reverts\n"
            "the morphological analysis from the original multilingual dataset\n"
            "to restore natural Japanese text without artificial spaces.\n"
        ),
        reference="https://huggingface.co/datasets/mteb/multilingual-sentiment-classification",
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["jpn-Jpan"],
        main_score="accuracy",
        date=("2022-08-01", "2022-08-01"),
        domains=["Reviews", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        adapted_from=["MultilingualSentimentClassification"],
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
    )
