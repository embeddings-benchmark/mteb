from __future__ import annotations

from mteb.abstasks.Audio.AbsTaskZeroshotAudioClassification import (
    AbsTaskZeroshotClassification,
)
from mteb.abstasks.TaskMetadata import TaskMetadata


class ESC50ZeroshotClassification(AbsTaskZeroshotClassification):
    metadata = TaskMetadata(
        name="ESC50ZeroShot",
        description="5-second clips of common environmental sounds, 50 classes",
        reference="https://dl.acm.org/doi/10.1145/2733373.2806390",
        dataset={
            "path": "karolpiczak/ESC-50"
        },
        type="ZeroShotClassification",
        category="a2t",
        eval_splits=["train"],
        eval_langs=["eng"],
        main_score="accuracy",
        domains=["Scene"],
        task_subtypes=["Scene recognition"],
        license="cc-by-nc-3.0",
        annotations_creators="human-annotated",
        modalities=["audio", "text"],
        sample_creation="found",
        bibtex_citation="""@inproceedings{piczak2015dataset,
        title = {{ESC}: {Dataset} for {Environmental Sound Classification}},
        author = {Piczak, Karol J.},
        booktitle = {Proceedings of the 23rd {Annual ACM Conference} on {Multimedia}},
        date = {2015-10-13},
        url = {http://dl.acm.org/citation.cfm?doid=2733373.2806390},
        doi = {10.1145/2733373.2806390},
        location = {{Brisbane, Australia}},
        isbn = {978-1-4503-3459-4},
        publisher = {{ACM Press}},
        pages = {1015--1018}
        }
        """,
        descriptive_stats={
            "n_samples": {"train": 2000}
        },
    )

    # Override default column name in the subclass
    audio_column_name: str = "audio"
    label_column_name: str = "category"

    def get_candidate_labels(self) -> list[str]:
        return [
            f"a sound of a {name}."
            for name in self.dataset["train"].features[self.label_column_name].names
        ]
