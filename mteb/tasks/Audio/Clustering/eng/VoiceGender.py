from __future__ import annotations

from mteb.abstasks.Audio.AbsTaskAudioClustering import AbsTaskAudioClustering
from mteb.abstasks.TaskMetadata import TaskMetadata


class VoiceGenderClustering(AbsTaskAudioClustering):
    label_column_name: str = "label"
    metadata = TaskMetadata(
        name="VoiceGenderClustering",
        description="Clustering audio recordings based on gender (male vs female).",
        reference="https://huggingface.co/datasets/mmn3690/voice-gender-clustering",
        dataset={
            "path": "mmn3690/voice-gender-clustering",
            "revision": "1b202ea7bcd0abd5283e628248803e1569257c80",
        },
        type="AudioClustering",
        category="a2a",
        eval_splits=["train"],
        eval_langs=["eng-Latn"],
        main_score="cluster_accuracy",
        date=("2024-01-01", "2024-12-31"),
        domains=["Spoken"],
        task_subtypes=["Gender Clustering"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        modalities=["audio"],
        sample_creation="found",
        bibtex_citation="""@inproceedings{Chung18b,
  author    = {Joon Son Chung and Arsha Nagrani and Andrew Zisserman},
  title     = {VoxCeleb2: Deep Speaker Recognition},
  booktitle = {Proceedings of Interspeech},
  year      = {2018}
}""",
    )
