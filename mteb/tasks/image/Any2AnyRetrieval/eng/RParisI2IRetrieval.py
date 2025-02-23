from __future__ import annotations

from mteb.abstasks.Image import AbsTaskAny2AnyRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class RParisEasyI2IRetrieval(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="RParisEasyI2IRetrieval",
        description="Retrieve photos of landmarks in Paris, France.",
        reference="https://openaccess.thecvf.com/content_cvpr_2018/html/Radenovic_Revisiting_Paris_and_CVPR_2018_paper.html",
        dataset={
            "path": "JamieSJS/r-paris-easy",
            "revision": "7d821ddebcb30ad343133e3a81e23347ac2a08a8",
        },
        type="Any2AnyRetrieval",
        category="i2i",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="cv_recall_at_1",
        date=("2009-01-01", "2010-04-01"),
        domains=["Web"],
        task_subtypes=["Object recognition"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        modalities=["image"],
        sample_creation="created",
        bibtex_citation="""@inproceedings{radenovic2018revisiting,
  title={Revisiting oxford and paris: Large-scale image retrieval benchmarking},
  author={Radenovi{\'c}, Filip and Iscen, Ahmet and Tolias, Giorgos and Avrithis, Yannis and Chum, Ond{\v{r}}ej},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={5706--5715},
  year={2018}
}
        """,
    )
    skip_first_result = False


class RParisMediumI2IRetrieval(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="RParisMediumI2IRetrieval",
        description="Retrieve photos of landmarks in Paris, France.",
        reference="https://openaccess.thecvf.com/content_cvpr_2018/html/Radenovic_Revisiting_Paris_and_CVPR_2018_paper.html",
        dataset={
            "path": "JamieSJS/r-paris-medium",
            "revision": "3d959815e102785efd628170281f1e65561b03d2",
        },
        type="Any2AnyRetrieval",
        category="i2i",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="cv_recall_at_1",
        date=("2009-01-01", "2010-04-01"),
        domains=["Web"],
        task_subtypes=["Object recognition"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        modalities=["image"],
        sample_creation="created",
        bibtex_citation="""@inproceedings{radenovic2018revisiting,
  title={Revisiting oxford and paris: Large-scale image retrieval benchmarking},
  author={Radenovi{\'c}, Filip and Iscen, Ahmet and Tolias, Giorgos and Avrithis, Yannis and Chum, Ond{\v{r}}ej},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={5706--5715},
  year={2018}
}
        """,
    )
    skip_first_result = False


class RParisHardI2IRetrieval(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="RParisHardI2IRetrieval",
        description="Retrieve photos of landmarks in Paris, France.",
        reference="https://openaccess.thecvf.com/content_cvpr_2018/html/Radenovic_Revisiting_Paris_and_CVPR_2018_paper.html",
        dataset={
            "path": "JamieSJS/r-paris-hard",
            "revision": "d3e0adf4e942446c04427511ccce281c86861248",
        },
        type="Any2AnyRetrieval",
        category="i2i",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="cv_recall_at_1",
        date=("2009-01-01", "2010-04-01"),
        domains=["Web"],
        task_subtypes=["Object recognition"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        modalities=["image"],
        sample_creation="created",
        bibtex_citation="""@inproceedings{radenovic2018revisiting,
  title={Revisiting oxford and paris: Large-scale image retrieval benchmarking},
  author={Radenovi{\'c}, Filip and Iscen, Ahmet and Tolias, Giorgos and Avrithis, Yannis and Chum, Ond{\v{r}}ej},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={5706--5715},
  year={2018}
}
        """,
    )
    skip_first_result = False
