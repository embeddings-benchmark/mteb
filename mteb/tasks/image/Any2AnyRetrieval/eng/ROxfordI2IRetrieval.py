from __future__ import annotations

from mteb.abstasks.Image import AbsTaskAny2AnyRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class ROxfordEasyI2IRetrieval(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="ROxfordEasyI2IRetrieval",
        description="Retrieve photos of landmarks in Oxford, UK.",
        reference="https://openaccess.thecvf.com/content_cvpr_2018/html/Radenovic_Revisiting_Oxford_and_CVPR_2018_paper.html",
        dataset={
            "path": "JamieSJS/r-oxford-easy",
            "revision": "b71b5f67a93aa63761b79a67bcf28bd2ae590902",
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


class ROxfordMediumI2IRetrieval(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="ROxfordMediumI2IRetrieval",
        description="Retrieve photos of landmarks in Oxford, UK.",
        reference="https://openaccess.thecvf.com/content_cvpr_2018/html/Radenovic_Revisiting_Oxford_and_CVPR_2018_paper.html",
        dataset={
            "path": "JamieSJS/r-oxford-medium",
            "revision": "1dfb86730ee4b3f49b441f4896d473c83eb5ff0d",
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


class ROxfordHardI2IRetrieval(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="ROxfordHardI2IRetrieval",
        description="Retrieve photos of landmarks in Oxford, UK.",
        reference="https://openaccess.thecvf.com/content_cvpr_2018/html/Radenovic_Revisiting_Oxford_and_CVPR_2018_paper.html",
        dataset={
            "path": "JamieSJS/r-oxford-hard",
            "revision": "f71ab9d4aabcda93d55a7e65edfb3a34767d89e6",
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
