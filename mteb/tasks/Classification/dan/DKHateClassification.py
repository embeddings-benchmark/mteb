from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class DKHateClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="DKHateClassification",
        dataset={
            "path": "DDSC/dkhate",
            "revision": "59d12749a3c91a186063c7d729ec392fda94681c",
        },
        description="Danish Tweets annotated for Hate Speech either being Offensive or not",
        reference="https://aclanthology.org/2020.lrec-1.430/",
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["dan-Latn"],
        main_score="accuracy",
        date=("2018-01-01", "2018-12-31"),
        form=["written"],
        domains=["Social"],
        task_subtypes=["Sentiment/Hate speech"],
        license="CC-BY-4.0",
        socioeconomic_status="mixed",
        annotations_creators="expert-annotated",
        dialect=[],
        text_creation="found",
        bibtex_citation="""@inproceedings{sigurbergsson-derczynski-2020-offensive,
    title = "Offensive Language and Hate Speech Detection for {D}anish",
    author = "Sigurbergsson, Gudbjartur Ingi  and
      Derczynski, Leon",
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
      Moreno, Asuncion  and
      Odijk, Jan  and
      Piperidis, Stelios",
    booktitle = "Proceedings of the Twelfth Language Resources and Evaluation Conference",
    month = may,
    year = "2020",
    address = "Marseille, France",
    publisher = "European Language Resources Association",
    url = "https://aclanthology.org/2020.lrec-1.430",
    pages = "3498--3508",
    abstract = "The presence of offensive language on social media platforms and the implications this poses is becoming a major concern in modern society. Given the enormous amount of content created every day, automatic methods are required to detect and deal with this type of content. Until now, most of the research has focused on solving the problem for the English language, while the problem is multilingual. We construct a Danish dataset DKhate containing user-generated comments from various social media platforms, and to our knowledge, the first of its kind, annotated for various types and target of offensive language. We develop four automatic classification systems, each designed to work for both the English and the Danish language. In the detection of offensive language in English, the best performing system achieves a macro averaged F1-score of 0.74, and the best performing system for Danish achieves a macro averaged F1-score of 0.70. In the detection of whether or not an offensive post is targeted, the best performing system for English achieves a macro averaged F1-score of 0.62, while the best performing system for Danish achieves a macro averaged F1-score of 0.73. Finally, in the detection of the target type in a targeted offensive post, the best performing system for English achieves a macro averaged F1-score of 0.56, and the best performing system for Danish achieves a macro averaged F1-score of 0.63. Our work for both the English and the Danish language captures the type and targets of offensive language, and present automatic methods for detecting different kinds of offensive language such as hate speech and cyberbullying.",
    language = "English",
    ISBN = "979-10-95546-34-4",
}""",
        n_samples={"test": 329},
        avg_character_length={"test": 104.0},
    )

    @property
    def metadata_dict(self) -> dict[str, str]:
        metadata_dict = dict(self.metadata)
        metadata_dict["n_experiments"] = 10
        metadata_dict["samples_per_label"] = 16
        return metadata_dict

    def dataset_transform(self):
        # convert label to a 0/1 label
        labels = self.dataset["train"]["label"]  # type: ignore
        lab2idx = {lab: idx for idx, lab in enumerate(set(labels))}
        self.dataset = self.dataset.map(
            lambda x: {"label": lab2idx[x["label"]]}, remove_columns=["label"]
        )
