from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata
from mteb.abstasks.zeroshot_classification import AbsTaskZeroShotClassification


class UCF101ZeroShotClassification(AbsTaskZeroShotClassification):
    metadata = TaskMetadata(
        name="UCF101ZeroShot",
        description="UCF101 is an action recognition data set of realistic action videos collected from YouTube, having 101 action categories. This version of the dataset does not contain images but images saved frame by frame. Train and test splits are generated based on the authors' first version train/test list.",
        reference="https://huggingface.co/datasets/flwrlabs/ucf101",
        dataset={
            "path": "mteb/ucf101",
            "revision": "e0618988874f6ffbf90af69fa6dccecb9be3deb3",
        },
        type="ZeroShotClassification",
        category="i2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=(
            "2012-01-01",
            "2012-12-01",
        ),  # Estimated range for the collection of reviews
        domains=["Activity", "Sport", "Web"],
        task_subtypes=["Activity recognition"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        modalities=["image", "text"],
        sample_creation="created",
        bibtex_citation=r"""
@misc{soomro2012ucf101dataset101human,
  archiveprefix = {arXiv},
  author = {Khurram Soomro and Amir Roshan Zamir and Mubarak Shah},
  eprint = {1212.0402},
  primaryclass = {cs.CV},
  title = {UCF101: A Dataset of 101 Human Actions Classes From Videos in The Wild},
  url = {https://arxiv.org/abs/1212.0402},
  year = {2012},
}
""",
    )

    def get_candidate_labels(self) -> list[str]:
        return [
            f"a photo of {name}"
            for name in self.dataset["test"].features[self.label_column_name].names
        ]


class UCF101VideoAudioZeroShotClassification(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="UCF101VideoAudioZeroShotClassification",
        description=(
            "Classifying video clips with audio into 51 human "
            "action categories from the UCF101 dataset. "
            "Used official split 1 restricted to a 51-class audio-bearing subset (~4,890 train / ~1,944 test)."
        ),
        reference="https://arxiv.org/abs/1212.0402",
        dataset={
            "path": "mteb/UCF101VideoAudioZeroShotClassification",
            "revision": "8f8c3417f6053179cf8d825c2f3ab293abf149fc",
        },
        type="VideoZeroshotClassification",
        category="va2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2012-01-01", "2012-12-03"),
        domains=["Activity", "Sport", "Web"],
        task_subtypes=["Activity recognition"],
        license="cc0-1.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video", "audio", "text"],
        sample_creation="found",
        bibtex_citation=r"""
@misc{soomro2012ucf101dataset101human,
  archiveprefix = {arXiv},
  author = {Khurram Soomro and Amir Roshan Zamir and Mubarak Shah},
  eprint = {1212.0402},
  primaryclass = {cs.CV},
  title = {UCF101: A Dataset of 101 Human Actions Classes From Videos in The Wild},
  url = {https://arxiv.org/abs/1212.0402},
  year = {2012},
}
""",
        is_beta=True,
    )
    max_fraction_of_documents_to_embed = None


class UCF101VideoZeroShotClassification(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="UCF101VideoZeroShotClassification",
        description=(
            "Classifying video clips with audio into 51 human "
            "action categories from the UCF101 dataset. "
            "Used official split 1 restricted to a 51-class audio-bearing subset (~4,890 train / ~1,944 test)."
        ),
        reference="https://arxiv.org/abs/1212.0402",
        dataset={
            "path": "mteb/UCF101VideoZeroShotClassification",
            "revision": "14a895f31b4d04f306c161b5ac127f386664d17c",
        },
        type="VideoZeroshotClassification",
        category="v2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2012-01-01", "2012-12-03"),
        domains=["Activity", "Sport", "Web"],
        task_subtypes=["Activity recognition"],
        license="cc0-1.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video", "text"],
        sample_creation="found",
        bibtex_citation=r"""
@misc{soomro2012ucf101dataset101human,
  archiveprefix = {arXiv},
  author = {Khurram Soomro and Amir Roshan Zamir and Mubarak Shah},
  eprint = {1212.0402},
  primaryclass = {cs.CV},
  title = {UCF101: A Dataset of 101 Human Actions Classes From Videos in The Wild},
  url = {https://arxiv.org/abs/1212.0402},
  year = {2012},
}
""",
    )
    max_fraction_of_documents_to_embed = None
