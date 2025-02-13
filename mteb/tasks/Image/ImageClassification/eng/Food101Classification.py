from __future__ import annotations

from mteb.abstasks.Image.AbsTaskImageClassification import AbsTaskImageClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class Food101Classification(AbsTaskImageClassification):
    metadata = TaskMetadata(
        name="Food101Classification",
        description="Classifying food.",
        reference="https://huggingface.co/datasets/ethz/food101",
        dataset={
            "path": "ethz/food101",
            "revision": "e06acf2a88084f04bce4d4a525165d68e0a36c38",
        },
        type="ImageClassification",
        category="i2i",
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
        bibtex_citation=""" @inproceedings{bossard14,
        title = {Food-101 -- Mining Discriminative Components with Random Forests},
        author = {Bossard, Lukas and Guillaumin, Matthieu and Van Gool, Luc},
        booktitle = {European Conference on Computer Vision},
        year = {2014}
        }
        """,
        descriptive_stats={
            "n_samples": {"validation": 25300},
            "avg_character_length": {"validation": 431.4},
        },
    )
