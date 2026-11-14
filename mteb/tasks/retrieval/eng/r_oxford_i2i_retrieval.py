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
            "path": "mteb/r-oxford-easy-multi",
            "revision": "063cf78b598275e0454311a329651eb708c6ccd7",
        },
        type="Any2AnyRetrieval",
        category="i2i",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="map_at_5",
        superseded_by="ROxfordEasyI2IRetrieval.v2",
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
            "path": "mteb/r-oxford-medium-multi",
            "revision": "b2f6a469e6d8126872049711e24c191f40698b04",
        },
        type="Any2AnyRetrieval",
        category="i2i",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="map_at_5",
        superseded_by="ROxfordMediumI2IRetrieval.v2",
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
            "path": "mteb/r-oxford-hard-multi",
            "revision": "d9d88e44ebba295dc660a2c7b1ce618a7e953375",
        },
        type="Any2AnyRetrieval",
        category="i2i",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="map_at_5",
        superseded_by="ROxfordHardI2IRetrieval.v2",
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


class ROxfordEasyI2IRetrievalV2(AbsTaskRetrieval):
    metadata = ROxfordEasyI2IRetrieval.metadata.model_copy(
        update={
            "name": "ROxfordEasyI2IRetrieval.v2",
            "description": "Retrieve photos of landmarks in Oxford, UK. Version 2 uses map_at_1000 as the main score: map_at_5 divides by all correct images, so with dozens to hundreds of correct images per query even a perfect ranking scores low.",
            "main_score": "map_at_1000",
            "adapted_from": ["ROxfordEasyI2IRetrieval"],
            "superseded_by": None,
        }
    )


class ROxfordMediumI2IRetrievalV2(AbsTaskRetrieval):
    metadata = ROxfordMediumI2IRetrieval.metadata.model_copy(
        update={
            "name": "ROxfordMediumI2IRetrieval.v2",
            "description": "Retrieve photos of landmarks in Oxford, UK. Version 2 uses map_at_1000 as the main score: map_at_5 divides by all correct images, so with dozens to hundreds of correct images per query even a perfect ranking scores low.",
            "main_score": "map_at_1000",
            "adapted_from": ["ROxfordMediumI2IRetrieval"],
            "superseded_by": None,
        }
    )


class ROxfordHardI2IRetrievalV2(AbsTaskRetrieval):
    metadata = ROxfordHardI2IRetrieval.metadata.model_copy(
        update={
            "name": "ROxfordHardI2IRetrieval.v2",
            "description": "Retrieve photos of landmarks in Oxford, UK. Version 2 uses map_at_1000 as the main score: map_at_5 divides by all correct images, so with dozens to hundreds of correct images per query even a perfect ranking scores low.",
            "main_score": "map_at_1000",
            "adapted_from": ["ROxfordHardI2IRetrieval"],
            "superseded_by": None,
        }
    )
