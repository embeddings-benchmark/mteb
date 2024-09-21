from __future__ import annotations

from mteb.abstasks.Image.AbsTaskImageClassification import AbsTaskImageClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class DTDClassification(AbsTaskImageClassification):
    metadata = TaskMetadata(
        name="DTD",
        description="Describable Textures Dataset in 47 categories.",
        reference="https://www.robots.ox.ac.uk/~vgg/data/dtd/",
        dataset={
            "path": "tanganke/dtd",
            "revision": "d2afa97d9f335b1a6b3b09c637aef667f98f966e",
        },
        type="Classification",
        category="i2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=(
            "2014-01-01",
            "2014-03-01",
        ),  # Estimated range for the collection of reviews
        domains=["Encyclopaedic"],
        task_subtypes=["Textures recognition"],
        license="Not specified",
        socioeconomic_status="mixed",
        annotations_creators="derived",
        dialect=[],
        modalities=["image"],
        sample_creation="created",
        bibtex_citation="""@InProceedings{cimpoi14describing,
            Author    = {M. Cimpoi and S. Maji and I. Kokkinos and S. Mohamed and and A. Vedaldi},
            Title     = {Describing Textures in the Wild},
            Booktitle = {Proceedings of the {IEEE} Conf. on Computer Vision and Pattern Recognition ({CVPR})},
            Year      = {2014}}
        """,
        descriptive_stats={
            "n_samples": {"test": 1880},
            "avg_character_length": {"test": 456},
        },
    )
