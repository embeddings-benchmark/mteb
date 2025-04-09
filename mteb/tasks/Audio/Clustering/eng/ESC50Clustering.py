from __future__ import annotations

from mteb.abstasks.Audio.AbsTaskAudioClustering import AbsTaskAudioClustering
from mteb.abstasks.TaskMetadata import TaskMetadata


class ESC50Clustering(AbsTaskAudioClustering):
    label_column_name: str = "target"
    metadata = TaskMetadata(
        name="ESC50Clustering",
        description="Clustering task based on the Environmental Sound Classification Dataset with 50 classes.",
        reference="https://huggingface.co/datasets/ashraq/esc50",
        dataset={
            "path": "ashraq/esc50",
            "revision": "e3e2a63ffff66b9a9735524551e3818e96af03ee",
        },
        type="AudioClustering",
        category="a2a",
        eval_splits=["train"],
        eval_langs=["eng-Latn"],
        main_score="cluster_accuracy",
        date=("2023-01-07", "2023-01-07"),
        domains=["Spoken", "Speech"],
        task_subtypes=["Environment Sound Clustering"],
        license="cc-by-nc-sa-3.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["audio"],
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
        }""",
    )
