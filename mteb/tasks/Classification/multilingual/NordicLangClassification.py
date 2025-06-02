from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class NordicLangClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="NordicLangClassification",
        description="A dataset for Nordic language identification.",
        reference="https://aclanthology.org/2021.vardial-1.8/",
        dataset={
            "path": "strombergnlp/nordic_langid",
            "revision": "e254179d18ab0165fdb6dbef91178266222bee2a",
            "name": "10k",
            "trust_remote_code": True,
        },
        type="Classification",
        category="s2s",
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
  abstract = {Automatic language identification is a challenging problem. Discriminating between closely related languages is especially difficult. This paper presents a machine learning approach for automatic language identification for the Nordic languages, which often suffer miscategorisation by existing state-of-the-art tools. Concretely we will focus on discrimination between six Nordic languages: Danish, Swedish, Norwegian (Nynorsk), Norwegian (Bokm{\aa}l), Faroese and Icelandic.},
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

    def dataset_transform(self):
        self.dataset = self.dataset.rename_columns(
            {"sentence": "text", "language": "label"}
        )
