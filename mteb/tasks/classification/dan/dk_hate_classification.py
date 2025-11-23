from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


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
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["dan-Latn"],
        main_score="accuracy",
        date=("2018-01-01", "2018-12-31"),
        domains=["Social", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{sigurbergsson-derczynski-2020-offensive,
  address = {Marseille, France},
  author = {Sigurbergsson, Gudbjartur Ingi  and
Derczynski, Leon},
  booktitle = {Proceedings of the Twelfth Language Resources and Evaluation Conference},
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
Moreno, Asuncion  and
Odijk, Jan  and
Piperidis, Stelios},
  isbn = {979-10-95546-34-4},
  language = {English},
  month = may,
  pages = {3498--3508},
  publisher = {European Language Resources Association},
  title = {Offensive Language and Hate Speech Detection for {D}anish},
  url = {https://aclanthology.org/2020.lrec-1.430},
  year = {2020},
}
""",
        prompt="Classify Danish tweets based on offensiveness (offensive, not offensive)",
        superseded_by="DKHateClassification.v2",
    )

    samples_per_label = 16

    def dataset_transform(self):
        # convert label to a 0/1 label
        labels = self.dataset["train"]["label"]  # type: ignore
        lab2idx = {lab: idx for idx, lab in enumerate(set(labels))}
        self.dataset = self.dataset.map(
            lambda x: {"label": lab2idx[x["label"]]}, remove_columns=["label"]
        )


class DKHateClassificationV2(AbsTaskClassification):
    metadata = TaskMetadata(
        name="DKHateClassification.v2",
        dataset={
            "path": "mteb/dk_hate",
            "revision": "0468ff11393992d8347cf4282fb706fe970608d4",
        },
        description="Danish Tweets annotated for Hate Speech either being Offensive or not This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)",
        reference="https://aclanthology.org/2020.lrec-1.430/",
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["dan-Latn"],
        main_score="accuracy",
        date=("2018-01-01", "2018-12-31"),
        domains=["Social", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{sigurbergsson-derczynski-2020-offensive,
  address = {Marseille, France},
  author = {Sigurbergsson, Gudbjartur Ingi  and
Derczynski, Leon},
  booktitle = {Proceedings of the Twelfth Language Resources and Evaluation Conference},
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
Moreno, Asuncion  and
Odijk, Jan  and
Piperidis, Stelios},
  isbn = {979-10-95546-34-4},
  language = {English},
  month = may,
  pages = {3498--3508},
  publisher = {European Language Resources Association},
  title = {Offensive Language and Hate Speech Detection for {D}anish},
  url = {https://aclanthology.org/2020.lrec-1.430},
  year = {2020},
}
""",
        prompt="Classify Danish tweets based on offensiveness (offensive, not offensive)",
        adapted_from=["DKHateClassification"],
    )

    samples_per_label = 16
