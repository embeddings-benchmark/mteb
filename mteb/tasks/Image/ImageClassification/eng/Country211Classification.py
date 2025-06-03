from __future__ import annotations

from mteb.abstasks.Image.AbsTaskImageClassification import AbsTaskImageClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class Country211Classification(AbsTaskImageClassification):
    metadata = TaskMetadata(
        name="Country211",
        description="Classifying images of 211 countries.",
        reference="https://huggingface.co/datasets/clip-benchmark/wds_country211",
        dataset={
            "path": "clip-benchmark/wds_country211",
            "revision": "1699f138f0558342a1cbf99f7cf36b4361bb5ebc",
        },
        type="ImageClassification",
        category="i2i",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=(
            "2020-01-01",
            "2021-02-26",
        ),  # Estimated range for the collection of reviews
        domains=["Scene"],
        task_subtypes=["Scene recognition"],
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        modalities=["image"],
        sample_creation="created",
        bibtex_citation=r"""
@article{radford2021learning,
  author = {Radford, Alec and Kim, Jong Wook and Hallacy, Chris and Ramesh, Aditya and Goh, Gabriel and Agarwal, Sandhini and Sastry, Girish and Askell, Amanda and Mishkin, Pamela and Clark, Jack and others},
  journal = {arXiv preprint arXiv:2103.00020},
  title = {Learning Transferable Visual Models From Natural Language Supervision},
  year = {2021},
}
""",
        descriptive_stats={
            "n_samples": {"test": 21100},
            "avg_character_length": {"test": 0},
        },
    )

    image_column_name: str = "jpg"
    label_column_name: str = "cls"
