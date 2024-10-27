from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


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
        category="s2s",
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
        bibtex_citation="""@inproceedings{quasthoff-etal-2006-corpus,
    title = "Corpus Portal for Search in Monolingual Corpora",
    author = "Quasthoff, Uwe  and
      Richter, Matthias  and
      Biemann, Christian",
    editor = "Calzolari, Nicoletta  and
      Choukri, Khalid  and
      Gangemi, Aldo  and
      Maegaard, Bente  and
      Mariani, Joseph  and
      Odijk, Jan  and
      Tapias, Daniel",
    booktitle = "Proceedings of the Fifth International Conference on Language Resources and Evaluation ({LREC}{'}06)",
    month = may,
    year = "2006",
    address = "Genoa, Italy",
    publisher = "European Language Resources Association (ELRA)",
    url = "http://www.lrec-conf.org/proceedings/lrec2006/pdf/641_pdf.pdf",
    abstract = "A simple and flexible schema for storing and presenting monolingual language resources is proposed. In this format, data for 18 different languages is already available in various sizes. The data is provided free of charge for online use and download. The main target is to ease the application of algorithms for monolingual and interlingual studies.",
}""",
        descriptive_stats={
            "n_samples": {"test": 150},
            "avg_character_length": {"test": 118.7},
        },
    )

    @property
    def metadata_dict(self) -> dict[str, str]:
        metadata_dict = super().metadata_dict
        metadata_dict["n_experiments"] = 10
        metadata_dict["samples_per_label"] = 16
        return metadata_dict
