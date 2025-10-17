import logging

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

logger = logging.getLogger(__name__)


class ROxfordEasyI2IRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="ROxfordEasyI2IRetrieval",
        description="Retrieve photos of landmarks in Oxford, UK.",
        reference="https://openaccess.thecvf.com/content_cvpr_2018/html/Radenovic_Revisiting_Oxford_and_CVPR_2018_paper.html",
        dataset={
            "path": "JamieSJS/r-oxford-easy-multi",
            "revision": "4c167c3ce529f19457c9b8e694258cc6cf8e7cc7",
        },
        type="Any2AnyRetrieval",
        category="i2i",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="map_at_5",
        date=("2009-01-01", "2010-04-01"),
        domains=["Web"],
        task_subtypes=["Object recognition"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        modalities=["image"],
        sample_creation="created",
        bibtex_citation=r"""
@inproceedings{radenovic2018revisiting,
  author = {Radenovi{\'c}, Filip and Iscen, Ahmet and Tolias, Giorgos and Avrithis, Yannis and Chum, Ond{\v{r}}ej},
  booktitle = {Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages = {5706--5715},
  title = {Revisiting oxford and paris: Large-scale image retrieval benchmarking},
  year = {2018},
}
""",
    )
    skip_first_result = False


class ROxfordMediumI2IRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="ROxfordMediumI2IRetrieval",
        description="Retrieve photos of landmarks in Oxford, UK.",
        reference="https://openaccess.thecvf.com/content_cvpr_2018/html/Radenovic_Revisiting_Oxford_and_CVPR_2018_paper.html",
        dataset={
            "path": "JamieSJS/r-oxford-medium-multi",
            "revision": "83bd440268e200a4f60313070618e3f45000fa94",
        },
        type="Any2AnyRetrieval",
        category="i2i",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="map_at_5",
        date=("2009-01-01", "2010-04-01"),
        domains=["Web"],
        task_subtypes=["Object recognition"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        modalities=["image"],
        sample_creation="created",
        bibtex_citation=r"""
@inproceedings{radenovic2018revisiting,
  author = {Radenovi{\'c}, Filip and Iscen, Ahmet and Tolias, Giorgos and Avrithis, Yannis and Chum, Ond{\v{r}}ej},
  booktitle = {Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages = {5706--5715},
  title = {Revisiting oxford and paris: Large-scale image retrieval benchmarking},
  year = {2018},
}
""",
    )
    skip_first_result = False


class ROxfordHardI2IRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="ROxfordHardI2IRetrieval",
        description="Retrieve photos of landmarks in Oxford, UK.",
        reference="https://openaccess.thecvf.com/content_cvpr_2018/html/Radenovic_Revisiting_Oxford_and_CVPR_2018_paper.html",
        dataset={
            "path": "JamieSJS/r-oxford-hard-multi",
            "revision": "fc7c4ae6655b1e6b132f3b262a359acef42dfce8",
        },
        type="Any2AnyRetrieval",
        category="i2i",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="map_at_5",
        date=("2009-01-01", "2010-04-01"),
        domains=["Web"],
        task_subtypes=["Object recognition"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        modalities=["image"],
        sample_creation="created",
        bibtex_citation=r"""
@inproceedings{radenovic2018revisiting,
  author = {Radenovi{\'c}, Filip and Iscen, Ahmet and Tolias, Giorgos and Avrithis, Yannis and Chum, Ond{\v{r}}ej},
  booktitle = {Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages = {5706--5715},
  title = {Revisiting oxford and paris: Large-scale image retrieval benchmarking},
  year = {2018},
}
""",
    )
    skip_first_result = False
