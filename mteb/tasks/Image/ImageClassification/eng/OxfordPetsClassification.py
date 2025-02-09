from __future__ import annotations

from mteb.abstasks.Image.AbsTaskImageClassification import AbsTaskImageClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class OxfordPetsClassification(AbsTaskImageClassification):
    metadata = TaskMetadata(
        name="OxfordPets",
        description="Classifying animal images.",
        reference="https://arxiv.org/abs/1306.5151",
        dataset={
            "path": "isaacchung/OxfordPets",
            "revision": "557b480fae8d69247be74d9503b378a09425096f",
        },
        type="ImageClassification",
        category="i2i",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=(
            "2009-01-01",
            "2010-01-01",
        ),  # Estimated range for the collection of reviews
        domains=["Encyclopaedic"],
        task_subtypes=["Object recognition"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        modalities=["image"],
        sample_creation="created",
        bibtex_citation="""@misc{maji2013finegrainedvisualclassificationaircraft,
            title={Fine-Grained Visual Classification of Aircraft}, 
            author={Subhransu Maji and Esa Rahtu and Juho Kannala and Matthew Blaschko and Andrea Vedaldi},
            year={2013},
            eprint={1306.5151},
            archivePrefix={arXiv},
            primaryClass={cs.CV},
            url={https://arxiv.org/abs/1306.5151}, 
        }
        """,
    )
