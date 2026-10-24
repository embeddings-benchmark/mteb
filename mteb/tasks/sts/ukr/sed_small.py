from mteb.abstasks.sts import AbsTaskSTS
from mteb.abstasks.task_metadata import TaskMetadata


class UkrSedUASmallSTSv1(AbsTaskSTS):
    column_names = ("query", "passage")
    metadata = TaskMetadata(
        name="UkrSedUASmallSTSv1",
        description="Small (100k+) synthetic dataset for fine-tuning text embedding models for Ukrainian language (STS task)",
        reference="https://huggingface.co/datasets/suntez13/sed-ua-small-sts-v1",
        type="STS",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["ukr-Cyrl"],
        main_score="cosine_spearman",
        dataset={
            "path": "mteb/UkrSedUASmallSTSv1",
            "revision": "d83d1c605ce02b53f03112349d7966ab692590b1",
        },
        date=("2025-03-24", "2025-03-24"),
        domains=["Constructed"],
        task_subtypes=[],
        license="bsd-3-clause",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@proceedings{SED-UA-small2025,
  author = {Oleksandr Mediakov and Dmytro Martjanov and Vasyl Lytvyn},
  booktitle = {Proceedings of the Information Systems and Networks (SISN), Volume 17},
  doi = {10.23939/sisn2025.17.403},
  pages = {403--410},
  publisher = {Lviv Polytechnic National University},
  title = {SED-UA-Small: Ukrainian Synthetic Dataset for Text Embedding Models},
  url = {https://science.lpnu.ua/sisn/all-volumes-and-issues/volume-17-2025/sed-ua-small-ukrainian-synthetic-dataset-text-embedding},
  year = {2025},
}
""",
    )
