from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class UANewsTitleClassification(AbsTaskClassification):
    input_column_name = "title"
    label_column_name = "labels"
    train_split = "train"
    metadata = TaskMetadata(
        name="UANewsTitleClassification",
        description="News classification into 5 classes (sports, news, politics, business, technology) via title column.",
        reference="https://huggingface.co/datasets/FIdo-AI/ua-news",
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["ukr-Cyrl"],
        main_score="accuracy",
        dataset={
            "path": "mteb/UANewsTitleClassification",
            "revision": "e0a56eb351f53b63a851c90ac13c531428d0e9e8",
        },
        date=("2020-01-01", "2022-07-05"),
        domains=["News"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""
@misc{huggingface_dataset_name,
  title = "FIdo-AI/ua-news",
  author = "FIdo-AI",
  year = {2022},
  howpublished = {https://huggingface.co/datasets/FIdo-AI/ua-news}
  }
""",
    )
