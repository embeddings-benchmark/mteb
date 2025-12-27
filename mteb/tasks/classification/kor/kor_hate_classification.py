from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class KorHateClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="KorHateClassification",
        description="The dataset was created to provide the first human-labeled Korean corpus for toxic speech detection from a Korean online entertainment news aggregator. Recently, two young Korean celebrities suffered from a series of tragic incidents that led to two major Korean web portals to close the comments section on their platform. However, this only serves as a temporary solution, and the fundamental issue has not been solved yet. This dataset hopes to improve Korean hate speech detection. Annotation was performed by 32 annotators, consisting of 29 annotators from the crowdsourcing platform DeepNatural AI and three NLP researchers.",
        dataset={
            "path": "mteb/KorHateClassification",
            "revision": "a4e70398c3689a5f55cd1f4a447d8d2da0a7dd1e",
        },
        reference="https://paperswithcode.com/dataset/korean-hatespeech-dataset",
        type="Classification",
        category="t2c",
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
        superseded_by="KorHateClassification.v2",
    )


class KorHateClassificationV2(AbsTaskClassification):
    metadata = TaskMetadata(
        name="KorHateClassification.v2",
        description="The dataset was created to provide the first human-labeled Korean corpus for toxic speech detection from a Korean online entertainment news aggregator. Recently, two young Korean celebrities suffered from a series of tragic incidents that led to two major Korean web portals to close the comments section on their platform. However, this only serves as a temporary solution, and the fundamental issue has not been solved yet. This dataset hopes to improve Korean hate speech detection. Annotation was performed by 32 annotators, consisting of 29 annotators from the crowdsourcing platform DeepNatural AI and three NLP researchers. This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)",
        dataset={
            "path": "mteb/kor_hate",
            "revision": "5d64e6dcbe9204c934e9a3852b1130a6f2d51ad4",
        },
        reference="https://paperswithcode.com/dataset/korean-hatespeech-dataset",
        type="Classification",
        category="t2c",
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
