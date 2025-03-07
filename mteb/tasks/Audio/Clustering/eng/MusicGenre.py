from __future__ import annotations

import librosa
import numpy as np
from datasets import Audio

from mteb.abstasks.Audio.AbsTaskAudioClustering import AbsTaskAudioClustering
from mteb.abstasks.TaskMetadata import TaskMetadata


class MusicGenreClustering(AbsTaskAudioClustering):
    label_column_name: str = "label"
    metadata = TaskMetadata(
        name="MusicGenreClustering",
        description="Clustering music recordings in 9 different genres.",
        reference="https://www-ai.cs.tu-dortmund.de/audio.html",
        dataset={
            "path": "mteb/music-genre",
            "revision": "2ed42a866b5155eb138eb3dc1e68515ccf3c8a50",
        },
        type="AudioClustering",
        category="a2a",
        eval_splits=["train"],
        eval_langs=["eng-Latn"],
        main_score="cluster_accuracy",
        date=("2005-01-01", "2005-12-31"),
        domains=["Music"],
        task_subtypes=["Music Clustering"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["audio"],
        sample_creation="found",
        bibtex_citation="""@inproceedings{homburg2005benchmark,
                        title={A Benchmark Dataset for Audio Classification and Clustering.},
                        author={Homburg, Helge and Mierswa, Ingo and M{\"o}ller, B{\"u}lent and Morik, Katharina and Wurst, Michael},
                        booktitle={ISMIR},
                        volume={2005},
                        pages={528--31},
                        year={2005}
                        }""",
    )

    def dataset_transform(self):
        self.dataset["train"] = self.dataset["train"].map(
            lambda example: {
                "audio": {
                    "array": np.array(librosa.load(example["audio"]["path"], sr=16000)[0]),
                    "sampling_rate": 16000,
                }
            }
        )
        self.dataset["train"] = self.dataset["train"].cast_column(
            "audio", Audio(sampling_rate=16000)
        )
