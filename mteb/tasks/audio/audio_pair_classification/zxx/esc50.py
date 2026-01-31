from mteb.abstasks import AbsTaskPairClassification
from mteb.abstasks.task_metadata import TaskMetadata


class ESC50PairClassification(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="ESC50PairClassification",
        description="Environmental Sound Classification Dataset.",
        reference="https://huggingface.co/datasets/ashraq/esc50",
        dataset={
            "path": "mteb/ESC50PairClassification",
            "revision": "0e610f3edcc1eb8f21e3806066e8462765c7a30a",
        },
        type="AudioPairClassification",
        category="a2a",
        eval_splits=["test"],
        eval_langs=["zxx-Zxxx"],
        main_score="max_ap",
        date=("2023-01-07", "2023-01-07"),
        domains=["Encyclopaedic"],
        task_subtypes=["Environment Sound Classification"],
        license="cc-by-nc-sa-3.0",
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

    input1_column_name: str = "audio1"
    input2_column_name: str = "audio2"
    label_column_name: str = "label"
