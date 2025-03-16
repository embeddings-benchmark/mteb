from __future__ import annotations

from mteb.abstasks.Audio.AbsTaskAudioClustering import AbsTaskAudioClustering
from mteb.abstasks.TaskMetadata import TaskMetadata


class VehicleSoundClustering(AbsTaskAudioClustering):
    metadata = TaskMetadata(
        name="VehicleSoundClustering",
        description="Clustering vehicle sounds recorded from smartphones (0 (car class), 1 (truck, bus and van class), 2 (motorcycle class))",
        reference="https://huggingface.co/datasets/DynamicSuperb/Vehicle_sounds_classification_dataset",
        dataset={
            "path": "DynamicSuperb/Vehicle_sounds_classification_dataset",
            "revision": "9ad231d349f9d0bddbf20a83d0d7635dccbd2501",
        },
        type="AudioClustering",
        category="a2a",
        eval_splits=["train"],
        eval_langs=["eng-Latn"],
        main_score="cluster_accuracy",
        date=("2024-06-20", "2024-06-20"),
        domains=["Scene"],
        task_subtypes=["Vehicle Clustering"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        modalities=["audio"],
        sample_creation="created",
        bibtex_citation="""@inproceedings{inproceedings,
            author = {Bazilinskyy, Pavlo and Aa, Arne and Schoustra, Michael and Spruit, John and Staats, Laurens and van der Vlist, Klaas Jan and de Winter, Joost},
            year = {2018},
            month = {05},
            pages = {},
            title = {An auditory dataset of passing vehicles recorded with a smartphone}
        }""",
        descriptive_stats={
            "n_samples": {"train": 1705},
        },
    )

    audio_column_name: str = "audio"
    label_column_name: str = "label"

    def dataset_transform(self):
        self.dataset['train'] = self.dataset.pop('test')
