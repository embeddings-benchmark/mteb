from __future__ import annotations

from mteb.abstasks.Audio.AbsTaskAudioMultilabelClassification import AbsTaskAudioMultilabelClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class ESC50Classification(AbsTaskAudioMultilabelClassification):
    
    metadata = TaskMetadata(
    name="ESC50_MultiLabel",
    description="Environmental Sound Classification Dataset.",
    reference="https://huggingface.co/datasets/ashraq/esc50",  # Replace with actual dataset URL
    dataset={
        "path": "ashraq/esc50",
        "revision": "e3e2a63ffff66b9a9735524551e3818e96af03ee",
    },
    type="AudioMultilabelClassification",
    category="a2a",
    eval_splits=["train"],
    eval_langs=["eng-Latn"],
    main_score="mAP",
    date=(
        "2023-01-07",
        "2023-01-07"
    ),
    domains=["Spoken"],  # Replace with appropriate domain from allowed list?? No appropriate domain name is available
    task_subtypes=["Environment Sound Classification"],
    license="cc-by-nc-sa-3.0",  # Replace with appropriate license from allowed list
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
    descriptive_stats={
        "n_samples": {"train": 2000},  # Need actual number
    },
)

    audio_column_name: str = "audio"
    label_column_name: str = "label"
    samples_per_label: int = 8  # Should we keep this the same as FSD50K?
