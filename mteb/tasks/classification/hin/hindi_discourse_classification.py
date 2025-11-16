from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class HindiDiscourseClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="HindiDiscourseClassification",
        dataset={
            "path": "mteb/HindiDiscourseClassification",
            "revision": "6f183d3e509464fd9d92516d4eff91e11b8ec622",
        },
        description="A Hindi Discourse dataset in Hindi with values for coherence.",
        reference="https://aclanthology.org/2020.lrec-1.149/",
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs=["hin-Deva"],
        main_score="accuracy",
        date=("2019-12-01", "2020-04-09"),
        domains=["Fiction", "Social", "Written"],
        dialect=[],
        task_subtypes=["Discourse coherence"],
        license="mit",
        annotations_creators="expert-annotated",
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{dhanwal-etal-2020-annotated,
  address = {Marseille, France},
  author = {Dhanwal, Swapnil  and
Dutta, Hritwik  and
Nankani, Hitesh  and
Shrivastava, Nilay  and
Kumar, Yaman  and
Li, Junyi Jessy  and
Mahata, Debanjan  and
Gosangi, Rakesh  and
Zhang, Haimin  and
Shah, Rajiv Ratn  and
Stent, Amanda},
  booktitle = {Proceedings of the 12th Language Resources and Evaluation Conference},
  isbn = {979-10-95546-34-4},
  language = {English},
  month = may,
  publisher = {European Language Resources Association},
  title = {An Annotated Dataset of Discourse Modes in {H}indi Stories},
  url = {https://www.aclweb.org/anthology/2020.lrec-1.149},
  year = {2020},
}
""",
        superseded_by="HindiDiscourseClassification.v2",
    )


class HindiDiscourseClassificationV2(AbsTaskClassification):
    metadata = TaskMetadata(
        name="HindiDiscourseClassification.v2",
        dataset={
            "path": "mteb/hindi_discourse",
            "revision": "9d10173a3df9858adc90711d8da9abf3df0a1259",
        },
        description="A Hindi Discourse dataset in Hindi with values for coherence. This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)",
        reference="https://aclanthology.org/2020.lrec-1.149/",
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["hin-Deva"],
        main_score="accuracy",
        date=("2019-12-01", "2020-04-09"),
        domains=["Fiction", "Social", "Written"],
        dialect=[],
        task_subtypes=["Discourse coherence"],
        license="mit",
        annotations_creators="expert-annotated",
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{dhanwal-etal-2020-annotated,
  address = {Marseille, France},
  author = {Dhanwal, Swapnil  and
Dutta, Hritwik  and
Nankani, Hitesh  and
Shrivastava, Nilay  and
Kumar, Yaman  and
Li, Junyi Jessy  and
Mahata, Debanjan  and
Gosangi, Rakesh  and
Zhang, Haimin  and
Shah, Rajiv Ratn  and
Stent, Amanda},
  booktitle = {Proceedings of the 12th Language Resources and Evaluation Conference},
  isbn = {979-10-95546-34-4},
  language = {English},
  month = may,
  publisher = {European Language Resources Association},
  title = {An Annotated Dataset of Discourse Modes in {H}indi Stories},
  url = {https://www.aclweb.org/anthology/2020.lrec-1.149},
  year = {2020},
}
""",
        adapted_from=["HindiDiscourseClassification"],
    )

    def dataset_transform(self):
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["train"]
        )
