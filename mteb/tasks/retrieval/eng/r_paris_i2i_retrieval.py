from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class RParisEasyI2IRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="RParisEasyI2IRetrieval",
        description="Retrieve photos of landmarks in Paris, UK.",
        reference="https://openaccess.thecvf.com/content_cvpr_2018/html/Radenovic_Revisiting_Paris_and_CVPR_2018_paper.html",
        dataset={
            "path": "mteb/r-paris-easy-multi",
            "revision": "868eac192da81e6b8e478f4d30bdb8b4a24e871b",
        },
        type="Any2AnyRetrieval",
        category="i2i",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="map_at_5",
        superseded_by="RParisEasyI2IRetrieval.v2",
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


class RParisMediumI2IRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="RParisMediumI2IRetrieval",
        description="Retrieve photos of landmarks in Paris, UK.",
        reference="https://openaccess.thecvf.com/content_cvpr_2018/html/Radenovic_Revisiting_Paris_and_CVPR_2018_paper.html",
        dataset={
            "path": "mteb/r-paris-medium-multi",
            "revision": "43d85ffeec4bacbfc9e1fe047e1eaf61c36fd8d6",
        },
        type="Any2AnyRetrieval",
        category="i2i",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="map_at_5",
        superseded_by="RParisMediumI2IRetrieval.v2",
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


class RParisHardI2IRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="RParisHardI2IRetrieval",
        description="Retrieve photos of landmarks in Paris, UK.",
        reference="https://openaccess.thecvf.com/content_cvpr_2018/html/Radenovic_Revisiting_Paris_and_CVPR_2018_paper.html",
        dataset={
            "path": "mteb/r-paris-hard-multi",
            "revision": "8a96e2c66a1ad6d0035df60accba4838c9d4e79f",
        },
        type="Any2AnyRetrieval",
        category="i2i",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="map_at_5",
        superseded_by="RParisHardI2IRetrieval.v2",
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


class RParisEasyI2IRetrievalV2(AbsTaskRetrieval):
    metadata = RParisEasyI2IRetrieval.metadata.model_copy(
        update={
            "name": "RParisEasyI2IRetrieval.v2",
            "description": "Retrieve photos of landmarks in Paris, France. Version 2 uses map_at_1000 as the main score: map_at_5 divides by all correct images, so with dozens to hundreds of correct images per query even a perfect ranking scores low.",
            "main_score": "map_at_1000",
            "adapted_from": ["RParisEasyI2IRetrieval"],
            "superseded_by": None,
        }
    )


class RParisMediumI2IRetrievalV2(AbsTaskRetrieval):
    metadata = RParisMediumI2IRetrieval.metadata.model_copy(
        update={
            "name": "RParisMediumI2IRetrieval.v2",
            "description": "Retrieve photos of landmarks in Paris, France. Version 2 uses map_at_1000 as the main score: map_at_5 divides by all correct images, so with dozens to hundreds of correct images per query even a perfect ranking scores low.",
            "main_score": "map_at_1000",
            "adapted_from": ["RParisMediumI2IRetrieval"],
            "superseded_by": None,
        }
    )


class RParisHardI2IRetrievalV2(AbsTaskRetrieval):
    metadata = RParisHardI2IRetrieval.metadata.model_copy(
        update={
            "name": "RParisHardI2IRetrieval.v2",
            "description": "Retrieve photos of landmarks in Paris, France. Version 2 uses map_at_1000 as the main score: map_at_5 divides by all correct images, so with dozens to hundreds of correct images per query even a perfect ranking scores low.",
            "main_score": "map_at_1000",
            "adapted_from": ["RParisHardI2IRetrieval"],
            "superseded_by": None,
        }
    )
