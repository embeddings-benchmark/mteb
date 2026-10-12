from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class ESC50ZeroshotClassification(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="ESC50_Zeroshot",
        description="Environmental Sound Classification Dataset.",
        reference="https://huggingface.co/datasets/ashraq/esc50",
        dataset={
            "path": "mteb/ESC50_Zeroshot",
            "revision": "5af9331bbd5a0747723d825659b7c613c56dbe9b",
        },
        type="AudioZeroshotClassification",
        category="a2t",
        eval_splits=["train"],
        eval_langs=["zxx-Zxxx"],
        main_score="accuracy",
        date=("2023-01-07", "2023-01-07"),
        domains=[
            "Spoken"
        ],  # Replace with appropriate domain from allowed list?? No appropriate domain name is available
        task_subtypes=["Environment Sound Classification"],
        license="cc-by-nc-sa-3.0",  # Replace with appropriate license from allowed list
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["audio", "text"],
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
