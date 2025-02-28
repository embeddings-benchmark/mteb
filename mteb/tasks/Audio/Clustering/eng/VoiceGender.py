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
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="clustering_accuracy",
        date=("2024-01-01", "2024-12-31"),
        domains=["Spoken"],
        task_subtypes=["Gender"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        modalities=["audio"],
        bibtex_citation="""@InProceedings{Chung18b,
              author       = "Chung, J.~S. and Nagrani, A. and Zisserman, A.",
              title        = "VoxCeleb2: Deep Speaker Recognition",
              booktitle    = "INTERSPEECH",
              year         = "2018
              }""",
    )
