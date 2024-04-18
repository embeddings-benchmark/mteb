from __future__ import annotations

from mteb.abstasks import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class CroatianSentimentClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="CroatianSentimentClassification",
        description="An Croatian dataset for sentiment classification.",
        reference="https://arxiv.org/abs/2009.08712",
        dataset={
            "path": "sepidmnorozy/Croatian_sentiment",
            "revision": "255da5c6b54c95faf74aba6d6cad9b2e176bf90a",
        },
        type="Classification",
        category="s2s",
        date=("2022-01-01", "2022-01-01"),
        eval_splits=["validation", "test"],
        eval_langs=["hrv-Latn"],
        main_score="accuracy",
        form=["written"],
        domains=["Reviews"],
        task_subtypes=["Sentiment/Hate speech"],
        license="Not specified",
        socioeconomic_status="mixed",
        annotations_creators="human-annotated",
        dialect=[],
        text_creation="found",
        bibtex_citation="""@inproceedings{mollanorozy-etal-2023-cross,
    title = "Cross-lingual Transfer Learning with {P}ersian",
    author = "Mollanorozy, Sepideh  and
      Tanti, Marc  and
      Nissim, Malvina",
    editor = "Beinborn, Lisa  and
      Goswami, Koustava  and
      Murado{\u{g}}lu, Saliha  and
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
    abstract = "The success of cross-lingual transfer learning for POS tagging has been shown to be strongly dependent, among other factors, on the (typological and/or genetic) similarity of the low-resource language used for testing and the language(s) used in pre-training or to fine-tune the model. We further unpack this finding in two directions by zooming in on a single language, namely Persian. First, still focusing on POS tagging we run an in-depth analysis of the behaviour of Persian with respect to closely related languages and languages that appear to benefit from cross-lingual transfer with Persian. To do so, we also use the World Atlas of Language Structures to determine which properties are shared between Persian and other languages included in the experiments. Based on our results, Persian seems to be a reasonable potential language for Kurmanji and Tagalog low-resource languages for other tasks as well. Second, we test whether previous findings also hold on a task other than POS tagging to pull apart the benefit of language similarity and the specific task for which such benefit has been shown to hold. We gather sentiment analysis datasets for 31 target languages and through a series of cross-lingual experiments analyse which languages most benefit from Persian as the source. The set of languages that benefit from Persian had very little overlap across the two tasks, suggesting a strong task-dependent component in the usefulness of language similarity in cross-lingual transfer.",
}
""",
        n_samples={"validation": 214, "test": 437},
        avg_character_length={"validation": 166.9, "test": 151.4},
    )
