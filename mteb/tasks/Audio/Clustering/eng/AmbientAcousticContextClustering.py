from __future__ import annotations

from mteb.abstasks.Audio.AbsTaskAudioClustering import AbsTaskAudioClustering
from mteb.abstasks.TaskMetadata import TaskMetadata


class AmbientAcousticContextClustering(AbsTaskAudioClustering):
    label_column_name: str = "label"
    metadata = TaskMetadata(
        name="AmbientAcousticContextClustering",
        description="Clustering task based on the Ambient Acoustic Context dataset containing 1-second segments for workplace activities.",
        reference="https://dl.acm.org/doi/10.1145/3379503.3403535",
        dataset={
            "path": "flwrlabs/ambient-acoustic-context",
            "revision": "8c77edafc0cad477055ec099c253c87b2b08e77a",
        },
        type="AudioClustering",
        category="a2a",
        eval_splits=["train"],
        eval_langs=["eng-Latn"],
        main_score="cluster_accuracy",
        date=("2020-01-01", "2020-12-31"),
        domains=["Spoken", "Speech"],
        task_subtypes=["Environment Sound Clustering"],
        license="unknown",
        annotations_creators="crowdsourced",
        dialect=[],
        modalities=["audio"],
        sample_creation="found",
        bibtex_citation="""@inproceedings{10.1145/3379503.3403535,
            author = {Park, Chunjong and Min, Chulhong and Bhattacharya, Sourav and Kawsar, Fahim},
            title = {Augmenting Conversational Agents with Ambient Acoustic Contexts},
            year = {2020},
            isbn = {9781450375160},
            publisher = {Association for Computing Machinery},
            address = {New York, NY, USA},
            url = {https://doi.org/10.1145/3379503.3403535},
            doi = {10.1145/3379503.3403535},
            booktitle = {22nd International Conference on Human-Computer Interaction with Mobile Devices and Services},
            articleno = {33},
            numpages = {9},
            keywords = {Acoustic ambient context, Conversational agents},
            location = {Oldenburg, Germany},
            series = {MobileHCI '20}
        }""",
    )
