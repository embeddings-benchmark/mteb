from __future__ import annotations

from mteb.abstasks.AbsTaskSTS import AbsTaskSTS
from mteb.abstasks.TaskMetadata import TaskMetadata


class SickrPLSTS(AbsTaskSTS):
    metadata = TaskMetadata(
        name="SICK-R-PL",
        dataset={
            "path": "PL-MTEB/sickr-pl-sts",
            "revision": "fd5c2441b7eeff8676768036142af4cfa42c1339",
        },
        description="Polish version of SICK dataset for textual relatedness.",
        reference="https://aclanthology.org/2020.lrec-1.207",
        type="STS",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["pol-Latn"],
        main_score="cosine_spearman",
        date=("2018-01-01", "2019-09-01"),  # rough estimate
        domains=["Web", "Written"],
        task_subtypes=["Textual Entailment"],
        license="cc-by-nc-sa-3.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="human-translated and localized",
        bibtex_citation="""
@inproceedings{dadas-etal-2020-evaluation,
    title = "Evaluation of Sentence Representations in {P}olish",
    author = "Dadas, Slawomir  and
      Perelkiewicz, Michal  and
      Poswiata, Rafal",
    editor = "Calzolari, Nicoletta  and
      B{\'e}chet, Fr{\'e}d{\'e}ric  and
      Blache, Philippe  and
      Choukri, Khalid  and
      Cieri, Christopher  and
      Declerck, Thierry  and
      Goggi, Sara  and
      Isahara, Hitoshi  and
      Maegaard, Bente  and
      Mariani, Joseph  and
      Mazo, Helene  and
      Moreno, Asuncion  and
      Odijk, Jan  and
      Piperidis, Stelios",
    booktitle = "Proceedings of the Twelfth Language Resources and Evaluation Conference",
    month = may,
    year = "2020",
    address = "Marseille, France",
    publisher = "European Language Resources Association",
    url = "https://aclanthology.org/2020.lrec-1.207",
    pages = "1674--1680",
    language = "English",
    ISBN = "979-10-95546-34-4",
}
        """,
    )

    @property
    def metadata_dict(self) -> dict[str, str]:
        metadata_dict = super().metadata_dict
        metadata_dict["min_score"] = 1
        metadata_dict["max_score"] = 5
        return metadata_dict


class CdscrSTS(AbsTaskSTS):
    metadata = TaskMetadata(
        name="CDSC-R",
        dataset={
            "path": "PL-MTEB/cdscr-sts",
            "revision": "1cd6abbb00df7d14be3dbd76a7dcc64b3a79a7cd",
        },
        description="Compositional Distributional Semantics Corpus for textual relatedness.",
        reference="https://aclanthology.org/P17-1073.pdf",
        type="STS",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["pol-Latn"],
        main_score="cosine_spearman",
        date=("2016-01-01", "2017-04-01"),  # rough estimate
        domains=["Web", "Written"],
        task_subtypes=["Textual Entailment"],
        license="cc-by-nc-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="human-translated and localized",
        bibtex_citation="""
@inproceedings{wroblewska-krasnowska-kieras-2017-polish,
    title = "{P}olish evaluation dataset for compositional distributional semantics models",
    author = "Wr{\'o}blewska, Alina  and
      Krasnowska-Kiera{\'s}, Katarzyna",
    editor = "Barzilay, Regina  and
      Kan, Min-Yen",
    booktitle = "Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2017",
    address = "Vancouver, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/P17-1073",
    doi = "10.18653/v1/P17-1073",
    pages = "784--792",
}

        """,
    )

    @property
    def metadata_dict(self) -> dict[str, str]:
        metadata_dict = super().metadata_dict
        metadata_dict["min_score"] = 1
        metadata_dict["max_score"] = 5
        return metadata_dict
