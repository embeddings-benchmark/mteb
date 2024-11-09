from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class GeorgianSentimentClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="GeorgianSentimentClassification",
        description="Goergian Sentiment Dataset",
        reference="https://aclanthology.org/2022.lrec-1.173",
        dataset={
            "path": "asparius/Georgian-Sentiment",
            "revision": "d4fb68dff38e89c42406080737b8431ea48fa866",
        },
        type="Classification",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["kat-Geor"],
        main_score="accuracy",
        date=("2022-01-01", "2022-06-25"),
        domains=["Reviews", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""
        @inproceedings{stefanovitch-etal-2022-resources,
    title = "Resources and Experiments on Sentiment Classification for {G}eorgian",
    author = "Stefanovitch, Nicolas  and
      Piskorski, Jakub  and
      Kharazi, Sopho",
    editor = "Calzolari, Nicoletta  and
      B{\'e}chet, Fr{\'e}d{\'e}ric  and
      Blache, Philippe  and
      Choukri, Khalid  and
      Cieri, Christopher  and
      Declerck, Thierry  and
      Goggi, Sara  and
      Isahara, Hitoshi  and
      Maegaard, Bente  and
      Mariani, Joseph  and
      Mazo, H{\'e}l{\`e}ne  and
      Odijk, Jan  and
      Piperidis, Stelios",
    booktitle = "Proceedings of the Thirteenth Language Resources and Evaluation Conference",
    month = jun,
    year = "2022",
    address = "Marseille, France",
    publisher = "European Language Resources Association",
    url = "https://aclanthology.org/2022.lrec-1.173",
    pages = "1613--1621",
    abstract = "This paper presents, to the best of our knowledge, the first ever publicly available annotated dataset for sentiment classification and semantic polarity dictionary for Georgian. The characteristics of these resources and the process of their creation are described in detail. The results of various experiments on the performance of both lexicon- and machine learning-based models for Georgian sentiment classification are also reported. Both 3-label (positive, neutral, negative) and 4-label settings (same labels + mixed) are considered. The machine learning models explored include, i.a., logistic regression, SVMs, and transformed-based models. We also explore transfer learning- and translation-based (to a well-supported language) approaches. The obtained results for Georgian are on par with the state-of-the-art results in sentiment classification for well studied languages when using training data of comparable size.",
}
        """,
    )
