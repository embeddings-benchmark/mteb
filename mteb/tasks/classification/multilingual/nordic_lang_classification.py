from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class NordicLangClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="NordicLangClassification",
        description="A dataset for Nordic language identification.",
        reference="https://aclanthology.org/2021.vardial-1.8/",
        dataset={
            "path": "mteb/NordicLangClassification",
            "revision": "425d7e9de276902b73d5359ecd4811f3b80cc7c0",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=[
            "nob-Latn",
            "nno-Latn",
            "dan-Latn",
            "swe-Latn",
            "isl-Latn",
            "fao-Latn",
        ],
        main_score="accuracy",
        date=("2020-01-01", "2021-12-31"),  # best guess, year of publication
        domains=["Encyclopaedic"],
        task_subtypes=["Language identification"],
        license="cc-by-sa-3.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{haas-derczynski-2021-discriminating,
  address = {Kiyv, Ukraine},
  author = {Haas, Ren{\'e}  and
Derczynski, Leon},
  booktitle = {Proceedings of the Eighth Workshop on NLP for Similar Languages, Varieties and Dialects},
  editor = {Zampieri, Marcos  and
Nakov, Preslav  and
Ljube{\v{s}}i{\'c}, Nikola  and
Tiedemann, J{\"o}rg  and
Scherrer, Yves  and
Jauhiainen, Tommi},
  month = apr,
  pages = {67--75},
  publisher = {Association for Computational Linguistics},
  title = {Discriminating Between Similar {N}ordic Languages},
  url = {https://aclanthology.org/2021.vardial-1.8},
  year = {2021},
}
""",
        prompt="Classify texts based on language",
    )

    samples_per_label = 32
