from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class Food101ZeroShotClassification(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Food101ZeroShot",
        description="Classifying food.",
        reference="https://huggingface.co/datasets/ethz/food101",
        dataset={
            "path": "mteb/Food101ZeroShot",
            "revision": "9cd17df6fbc36afc45cbadf47ae474365a8ae623",
        },
        type="ZeroShotClassification",
        category="i2t",
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
        modalities=["text", "image"],
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
