from mteb.abstasks.audio.abs_task_audio_classification import (
    AbsTaskAudioClassification,
)
from mteb.abstasks.task_metadata import TaskMetadata


class ESC50Classification(AbsTaskAudioClassification):
    metadata = TaskMetadata(
        name="ESC50",
        description="Environmental Sound Classification Dataset.",
        reference="https://huggingface.co/datasets/ashraq/esc50",
        dataset={
            "path": "mteb/esc50",
            "revision": "0877d3708afd4deba0b89a2b54eb528e7f24ec6c",
        },
        type="AudioClassification",
        category="a2t",
        eval_splits=["train"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2023-01-07", "2023-01-07"),
        domains=[
            "Spoken"
        ],  # Replace with appropriate domain from allowed list?? No appropriate domain name is available
        task_subtypes=["Environment Sound Classification"],
        license="cc-by-nc-sa-3.0",  # Replace with appropriate license from allowed list
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["audio"],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{piczak2015dataset,
  author = {Piczak, Karol J.},
  booktitle = {Proceedings of the 23rd {Annual ACM Conference} on {Multimedia}},
  date = {2015-10-13},
  doi = {10.1145/2733373.2806390},
  isbn = {978-1-4503-3459-4},
  location = {{Brisbane, Australia}},
  pages = {1015--1018},
  publisher = {{ACM Press}},
  title = {{ESC}: {Dataset} for {Environmental Sound Classification}},
  url = {http://dl.acm.org/citation.cfm?doid=2733373.2806390},
}
""",
    )

    audio_column_name: str = "audio"
    label_column_name: str = "target"
    samples_per_label: int = 50
    is_cross_validation: bool = True
