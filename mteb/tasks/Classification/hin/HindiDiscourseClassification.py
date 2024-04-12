from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class HindiDiscourseClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="HindiDiscourseClassification",
        dataset={
            "path": "hindi_discourse",
            "revision": "218ce687943a0da435d6d62751a4ab216be6cd40",
        },
        description="A Hindi Discourse dataset in Hindi with values for coherence.",
        reference="https://aclanthology.org/2020.lrec-1.149/",
        type="Classification",
        category="s2s",
        eval_splits=["train"],
        eval_langs=["hin"],
        main_score="accuracy",
        date=("2019-12-01", "2020-04-09"),
        form=["written"],
        domains=["Fiction", "Social"],
        dialect=[],
        task_subtypes=["Discourse coherence"],
        license="MIT",
        socioeconomic_status="medium",
        annotations_creators="expert-annotated",
        text_creation="found",
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
    pages = "1191--1196",
    abstract = "In this paper, we present a new corpus consisting of sentences from Hindi short stories annotated for five different discourse modes argumentative, narrative, descriptive, dialogic and informative. We present a detailed account of the entire data collection and annotation processes. The annotations have a very high inter-annotator agreement (0.87 k-alpha). We analyze the data in terms of label distributions, part of speech tags, and sentence lengths. We characterize the performance of various classification algorithms on this dataset and perform ablation studies to understand the nature of the linguistic models suitable for capturing the nuances of the embedded discourse structures in the presented corpus.",
    language = "English",
    ISBN = "979-10-95546-34-4",
}""",
        n_samples={"train": 2048},
        avg_character_length=None,
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_columns(
            {"Sentence": "text", "Discourse Mode": "label"}
        ).remove_columns(["Story_no"])
        self.dataset["train"] = self.dataset["train"].select(range(2048))
