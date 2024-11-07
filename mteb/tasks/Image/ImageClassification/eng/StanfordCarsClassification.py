from __future__ import annotations

from mteb.abstasks.Image.AbsTaskImageClassification import AbsTaskImageClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class StanfordCarsClassification(AbsTaskImageClassification):
    metadata = TaskMetadata(
        name="StanfordCars",
        description="Classifying car images from 196 makes.",
        reference="https://pure.mpg.de/rest/items/item_2029263/component/file_2029262/content",
        dataset={
            "path": "isaacchung/StanfordCars",
            "revision": "09ffe9bc7864d3f1e851529e5c4b7e05601a04fb",
        },
        type="ImageClassification",
        category="i2i",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=(
            "2013-01-01",
            "2013-04-01",
        ),  # Estimated range for the collection of reviews
        domains=["Encyclopaedic"],
        task_subtypes=["Object recognition"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        modalities=["image"],
        sample_creation="created",
        bibtex_citation="""@inproceedings{Krause2013CollectingAL,
        title={Collecting a Large-scale Dataset of Fine-grained Cars},
        author={Jonathan Krause and Jia Deng and Michael Stark and Li Fei-Fei},
        year={2013},
        url={https://api.semanticscholar.org/CorpusID:16632981}
        }
        """,
        descriptive_stats={
            "n_samples": {"test": 8041},
            "avg_character_length": {"test": 431.4},
        },
    )
