from __future__ import annotations

from mteb.abstasks.Audio.AbsTaskZeroshotAudioClassification import (
    AbsTaskZeroshotClassification,
)
from mteb.abstasks.TaskMetadata import TaskMetadata


class UrbanSound8kZeroshotClassification(AbsTaskZeroshotClassification):
    metadata = TaskMetadata(
        name="UrbanSound8kZeroshot",
        description="Classifying 4s urban sounds into 10 classes",
        reference="https://dl.acm.org/doi/10.1145/2647868.2655045",
        dataset={
            "path": "danavery/urbansound8K"
        },
        type="ZeroShotClassification",
        category="a2t",
        eval_splits=["train"],
        eval_langs=["eng-latn"],
        main_score="accuracy",
        domains=["Scene"],
        task_subtypes=["Scene recognition"],
        license="cc-by-nc-4.0",
        modalities=["audio", "text"],
        sample_creation="found",
        bibtex_citation="""@inproceedings{10.1145/2647868.2655045,
        author = {Salamon, Justin and Jacoby, Christopher and Bello, Juan Pablo},
        title = {A Dataset and Taxonomy for Urban Sound Research},
        year = {2014},
        isbn = {9781450330633},
        publisher = {Association for Computing Machinery},
        address = {New York, NY, USA},
        url = {https://doi.org/10.1145/2647868.2655045},
        doi = {10.1145/2647868.2655045},
        abstract = {Automatic urban sound classification is a growing area of research with applications in multimedia retrieval and urban informatics. In this paper we identify two main barriers to research in this area - the lack of a common taxonomy and the scarceness of large, real-world, annotated data. To address these issues we present a taxonomy of urban sounds and a new dataset, UrbanSound, containing 27 hours of audio with 18.5 hours of annotated sound event occurrences across 10 sound classes. The challenges presented by the new dataset are studied through a series of experiments using a baseline classification system.},
        ooktitle = {Proceedings of the 22nd ACM International Conference on Multimedia},
        pages = {1041–1044},
        numpages = {4},
        keywords = {classification, dataset, taxonomy, urban sound},
        location = {Orlando, Florida, USA},
        series = {MM '14}
        }
        """,
        descriptive_stats={
            "n_samples": {"train": 8730}
        },
    )

    # Override default column name in the subclass
    audio_column_name: str = "audio"
    label_column_name: str = "class"

    def get_candidate_labels(self) -> list[str]:
        """
        Returns a list of formatted candidate labels based on unique sound classes in the dataset.
        """
        unique_classes = set(example[self.label_column_name] for example in self.dataset["train"])

        return [f"a sound of a {name}." for name in unique_classes]
