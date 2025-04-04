from __future__ import annotations

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
            "revision": "eb78950d473b87812357c207bae9c36779383e09",
        },
        type="AudioClustering",
        category="a2a",
        eval_splits=["test"],
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
