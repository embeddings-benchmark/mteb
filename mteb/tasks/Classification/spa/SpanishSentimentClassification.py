from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
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
        license="not specified",
        annotations_creators="derived",
        dialect=[],
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
    )
