from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


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
        category="t2c",
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
        bibtex_citation=r"""
@inproceedings{stefanovitch-etal-2022-resources,
  address = {Marseille, France},
  author = {Stefanovitch, Nicolas  and
Piskorski, Jakub  and
Kharazi, Sopho},
  booktitle = {Proceedings of the Thirteenth Language Resources and Evaluation Conference},
  editor = {Calzolari, Nicoletta  and
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
Piperidis, Stelios},
  month = jun,
  pages = {1613--1621},
  publisher = {European Language Resources Association},
  title = {Resources and Experiments on Sentiment Classification for {G}eorgian},
  url = {https://aclanthology.org/2022.lrec-1.173},
  year = {2022},
}
""",
    )
