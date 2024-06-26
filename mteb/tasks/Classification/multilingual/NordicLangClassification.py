from __future__ import annotations

from mteb.abstasks import AbsTaskClassification
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
        date=None,
        form=None,
        domains=None,
        task_subtypes=None,
        license=None,
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=None,
        text_creation=None,
        bibtex_citation="""@inproceedings{haas-derczynski-2021-discriminating,
    title = "Discriminating Between Similar {N}ordic Languages",
    author = "Haas, Ren{\'e}  and
      Derczynski, Leon",
    editor = {Zampieri, Marcos  and
      Nakov, Preslav  and
      Ljube{\v{s}}i{\'c}, Nikola  and
      Tiedemann, J{\"o}rg  and
      Scherrer, Yves  and
      Jauhiainen, Tommi},
    booktitle = "Proceedings of the Eighth Workshop on NLP for Similar Languages, Varieties and Dialects",
    month = apr,
    year = "2021",
    address = "Kiyv, Ukraine",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.vardial-1.8",
    pages = "67--75",
    abstract = "Automatic language identification is a challenging problem. Discriminating between closely related languages is especially difficult. This paper presents a machine learning approach for automatic language identification for the Nordic languages, which often suffer miscategorisation by existing state-of-the-art tools. Concretely we will focus on discrimination between six Nordic languages: Danish, Swedish, Norwegian (Nynorsk), Norwegian (Bokm{\aa}l), Faroese and Icelandic.",
}
""",
        n_samples={"test": 3000},
        avg_character_length={"test": 78.2},
    )

    @property
    def metadata_dict(self) -> dict[str, str]:
        metadata_dict = super().metadata_dict
        metadata_dict["n_experiments"] = 10
        metadata_dict["samples_per_label"] = 32
        return metadata_dict

    def dataset_transform(self):
        self.dataset = self.dataset.rename_columns(
            {"sentence": "text", "language": "label"}
        )
