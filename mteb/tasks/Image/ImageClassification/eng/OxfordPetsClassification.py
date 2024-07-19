from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from .....abstasks import AbsTaskImageClassification

## mteb run -m openai/clip-vit-base-patch32 --model_revision 3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268 --output_folder results-mieb -t OxfordPets


class OxfordPetsClassification(AbsTaskImageClassification):
    metadata = TaskMetadata(
        name="OxfordPets",
        description="Classifying animal images.",
        reference="https://arxiv.org/abs/1306.5151",
        dataset={
            "path": "isaacchung/OxfordPets",
            "revision": "557b480fae8d69247be74d9503b378a09425096f",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=(
            "2009-01-01",
            "2010-01-01",
        ),  # Estimated range for the collection of reviews
        domains=["Encyclopaedic"],
        task_subtypes=["Object recognition"],
        license="Not specified",
        socioeconomic_status="mixed",
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
        descriptive_stats={
            "n_samples": {"test": 3669},
            "avg_character_length": {"test": 431.4},
        },
    )
