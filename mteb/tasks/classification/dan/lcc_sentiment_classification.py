from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class LccSentimentClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="LccSentimentClassification",
        dataset={
            "path": "DDSC/lcc",
            "revision": "de7ba3406ee55ea2cc52a0a41408fa6aede6d3c6",
        },
        description="The leipzig corpora collection, annotated for sentiment",
        reference="https://github.com/fnielsen/lcc-sentiment",
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["dan-Latn"],
        main_score="accuracy",
        date=("2006-01-01", "2006-12-31"),
        domains=["News", "Web", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{quasthoff-etal-2006-corpus,
  address = {Genoa, Italy},
  author = {Quasthoff, Uwe  and
Richter, Matthias  and
Biemann, Christian},
  booktitle = {Proceedings of the Fifth International Conference on Language Resources and Evaluation ({LREC}{'}06)},
  editor = {Calzolari, Nicoletta  and
Choukri, Khalid  and
Gangemi, Aldo  and
Maegaard, Bente  and
Mariani, Joseph  and
Odijk, Jan  and
Tapias, Daniel},
  month = may,
  publisher = {European Language Resources Association (ELRA)},
  title = {Corpus Portal for Search in Monolingual Corpora},
  url = {http://www.lrec-conf.org/proceedings/lrec2006/pdf/641_pdf.pdf},
  year = {2006},
}
""",
        prompt="Classify texts based on sentiment",
    )

    samples_per_label = 16
