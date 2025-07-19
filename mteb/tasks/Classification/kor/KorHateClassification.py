from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class KorHateClassification(AbsTaskClassification):
    superseded_by = "KorHateClassification.v2"
    metadata = TaskMetadata(
        name="KorHateClassification",
        description="""The dataset was created to provide the first human-labeled Korean corpus for
        toxic speech detection from a Korean online entertainment news aggregator. Recently,
        two young Korean celebrities suffered from a series of tragic incidents that led to two
        major Korean web portals to close the comments section on their platform. However, this only
        serves as a temporary solution, and the fundamental issue has not been solved yet. This dataset
        hopes to improve Korean hate speech detection. Annotation was performed by 32 annotators,
        consisting of 29 annotators from the crowdsourcing platform DeepNatural AI and three NLP researchers.
        """,
        dataset={
            "path": "inmoonlight/kor_hate",
            "revision": "bd1a7370caf712125fac1fda375834ca8ddefaca",
            "trust_remote_code": True,
        },
        reference="https://paperswithcode.com/dataset/korean-hatespeech-dataset",
        type="Classification",
        category="s2s",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs=["kor-Hang"],
        main_score="accuracy",
        date=("2018-01-01", "2020-01-01"),
        domains=["Social", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="cc-by-sa-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@misc{moon2020beep,
  archiveprefix = {arXiv},
  author = {Jihyung Moon and Won Ik Cho and Junbum Lee},
  eprint = {2005.12503},
  primaryclass = {cs.CL},
  title = {BEEP! Korean Corpus of Online News Comments for Toxic Speech Detection},
  year = {2020},
}
""",
    )

    def dataset_transform(self):
        keep_cols = ["comments", "hate"]
        rename_dict = dict(zip(keep_cols, ["text", "label"]))
        remove_cols = [
            col for col in self.dataset["test"].column_names if col not in keep_cols
        ]
        self.dataset = self.dataset.rename_columns(rename_dict)
        self.dataset = self.dataset.remove_columns(remove_cols)
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["train"]
        )


class KorHateClassificationV2(AbsTaskClassification):
    metadata = TaskMetadata(
        name="KorHateClassification.v2",
        description="""The dataset was created to provide the first human-labeled Korean corpus for
        toxic speech detection from a Korean online entertainment news aggregator. Recently,
        two young Korean celebrities suffered from a series of tragic incidents that led to two
        major Korean web portals to close the comments section on their platform. However, this only
        serves as a temporary solution, and the fundamental issue has not been solved yet. This dataset
        hopes to improve Korean hate speech detection. Annotation was performed by 32 annotators,
        consisting of 29 annotators from the crowdsourcing platform DeepNatural AI and three NLP researchers.

        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)""",
        dataset={
            "path": "mteb/kor_hate",
            "revision": "5d64e6dcbe9204c934e9a3852b1130a6f2d51ad4",
        },
        reference="https://paperswithcode.com/dataset/korean-hatespeech-dataset",
        type="Classification",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["kor-Hang"],
        main_score="accuracy",
        date=("2018-01-01", "2020-01-01"),
        domains=["Social", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="cc-by-sa-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@misc{moon2020beep,
  archiveprefix = {arXiv},
  author = {Jihyung Moon and Won Ik Cho and Junbum Lee},
  eprint = {2005.12503},
  primaryclass = {cs.CL},
  title = {BEEP! Korean Corpus of Online News Comments for Toxic Speech Detection},
  year = {2020},
}
""",
        adapted_from=["KorHateClassification"],
    )

    def dataset_transform(self):
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["train"]
        )
