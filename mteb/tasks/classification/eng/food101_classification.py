from mteb.abstasks import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class Food101Classification(AbsTaskClassification):
    input_column_name: str = "image"
    samples_per_label: int = 16
    n_experiments: int = 5

    metadata = TaskMetadata(
        name="Food101Classification",
        description="Classifying food.",
        reference="https://huggingface.co/datasets/ethz/food101",
        dataset={
            "path": "ethz/food101",
            "revision": "e06acf2a88084f04bce4d4a525165d68e0a36c38",
        },
        type="ImageClassification",
        category="i2c",
        eval_splits=["validation"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=(
            "2013-01-01",
            "2014-01-01",
        ),  # Estimated range for the collection of reviews
        domains=["Web"],
        task_subtypes=["Object recognition"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        modalities=["image"],
        sample_creation="created",
        bibtex_citation=r"""
@inproceedings{bossard14,
  author = {Bossard, Lukas and Guillaumin, Matthieu and Van Gool, Luc},
  booktitle = {European Conference on Computer Vision},
  title = {Food-101 -- Mining Discriminative Components with Random Forests},
  year = {2014},
}
""",
    )
