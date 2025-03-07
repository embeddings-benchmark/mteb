from __future__ import annotations

from mteb.abstasks.Audio.AbsTaskZeroshotAudioClassification import (
    AbsTaskZeroshotAudioClassification,
)
from mteb.abstasks.TaskMetadata import TaskMetadata


class ESC50ZeroshotClassification(AbsTaskZeroshotAudioClassification):
    metadata = TaskMetadata(
        name="ESC50ZeroShot",
        description="5-second clips of common environmental sounds, 50 classes",
        reference="https://dl.acm.org/doi/10.1145/2733373.2806390",
        dataset={"path": "ashraq/esc50", "revision": "5c72356dbaaa04826a28c94283a15b112ddeca02"},
        type="ZeroShotClassification",
        category="t2t", #not actually t2t, put it to avoid errors
        eval_splits=["train"],
        eval_langs=["eng-latn"],
        main_score="accuracy",
        domains=["Scene"],
        task_subtypes=["Scene recognition"],
        license="cc-by-nc-4.0",
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
        descriptive_stats={"n_samples": {"train": 2000}},
    )

    # Override default column name in the subclass
    audio_column_name: str = "audio"
    label_column_name: str = "category"
    target_column_name: str = "target"

    def get_candidate_labels(self) -> list[str]:
        pairs = set()
        for row in self.dataset["train"]:
            pairs.add((row[self.target_column_name], row[self.label_column_name]))
        
        print(sorted(list(pairs)))
        return [
            f"a sound of a {name}."
            for _, name in sorted(list(pairs))
        ]
