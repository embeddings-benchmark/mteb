from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class HindiDiscourseClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="HindiDiscourseClassification",
        dataset={
            "path": "midas/hindi_discourse",
            "revision": "218ce687943a0da435d6d62751a4ab216be6cd40",
            "trust_remote_code": True,
        },
        description="A Hindi Discourse dataset in Hindi with values for coherence.",
        reference="https://aclanthology.org/2020.lrec-1.149/",
        type="Classification",
        category="s2s",
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
        bibtex_citation="""
        @inproceedings{dhanwal-etal-2020-annotated,
    title = "An Annotated Dataset of Discourse Modes in {H}indi Stories",
    author = "Dhanwal, Swapnil  and
      Dutta, Hritwik  and
      Nankani, Hitesh  and
      Shrivastava, Nilay  and
      Kumar, Yaman  and
      Li, Junyi Jessy  and
      Mahata, Debanjan  and
      Gosangi, Rakesh  and
      Zhang, Haimin  and
      Shah, Rajiv Ratn  and
      Stent, Amanda",
    booktitle = "Proceedings of the 12th Language Resources and Evaluation Conference",
    month = may,
    year = "2020",
    address = "Marseille, France",
    publisher = "European Language Resources Association",
    url = "https://www.aclweb.org/anthology/2020.lrec-1.149",
    language = "English",
    ISBN = "979-10-95546-34-4",
}""",
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_columns(
            {"Sentence": "text", "Discourse Mode": "label"}
        ).remove_columns(["Story_no"])
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["train"]
        )
