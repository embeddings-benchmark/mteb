from __future__ import annotations

from mteb.abstasks.Audio.AbsTaskAudioClustering import AbsTaskAudioClustering
from mteb.abstasks.TaskMetadata import TaskMetadata


class GTZANGenreClustering(AbsTaskAudioClustering):
    label_column_name: str = "label"
    metadata = TaskMetadata(
        name="GTZANGenreClustering",
        description="Music genre clustering task based on GTZAN dataset with 10 music genres.",
        reference="https://huggingface.co/datasets/silky1708/GTZAN-Genre",
        dataset={
            "path": "silky1708/GTZAN-Genre",
            "revision": "5efdda59d0d185bfe17ada9b54d233349d0e0168",
        },
        type="AudioClustering",
        category="a2a",
        eval_splits=["train"],
        eval_langs=["eng-Latn"],
        main_score="cluster_accuracy",
        date=("2000-01-01", "2001-12-31"),
        domains=["Music"],
        task_subtypes=["Music Clustering"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["audio"],
        sample_creation="found",
        bibtex_citation="""@ARTICLE{1021072,
            author={Tzanetakis, G. and Cook, P.},
            journal={IEEE Transactions on Speech and Audio Processing}, 
            title={Musical genre classification of audio signals}, 
            year={2002},
            volume={10},
            number={5},
            pages={293-302},
            keywords={Humans;Music information retrieval;Instruments;Computer science;Multiple signal classification;Signal analysis;Pattern recognition;Feature extraction;Wavelet analysis;Cultural differences},
            doi={10.1109/TSA.2002.800560}}""",
    ) 