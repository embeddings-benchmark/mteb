from __future__ import annotations

from mteb.abstasks.Image.AbsTaskAny2AnyMultiChoice import MultiChoiceEvaluationMixin
from mteb.abstasks.Image.AbsTaskAny2AnyRetrieval import AbsTaskAny2AnyRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


class RParisEasyI2IRetrieval(MultiChoiceEvaluationMixin, AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="RParisEasyI2IRetrieval",
        description="Retrieve photos of landmarks in Paris, UK.",
        reference="https://openaccess.thecvf.com/content_cvpr_2018/html/Radenovic_Revisiting_Paris_and_CVPR_2018_paper.html",
        dataset={
            "path": "JamieSJS/r-paris-easy-multi",
            "revision": "db94b5afd0014ab8c978f20a0fbcc52da1612a08",
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
        descriptive_stats={
            "n_samples": {"test": 6392},
            "avg_character_length": {
                "test": {
                    "average_document_length": 0.0,
                    "average_query_length": 0.0,
                    "num_documents": 6322,
                    "num_queries": 70,
                    "average_relevant_docs_per_query": 98.2,
                }
            },
        },
    )
    skip_first_result = False


class RParisMediumI2IRetrieval(MultiChoiceEvaluationMixin, AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="RParisMediumI2IRetrieval",
        description="Retrieve photos of landmarks in Paris, UK.",
        reference="https://openaccess.thecvf.com/content_cvpr_2018/html/Radenovic_Revisiting_Paris_and_CVPR_2018_paper.html",
        dataset={
            "path": "JamieSJS/r-paris-medium-multi",
            "revision": "372c79fc823e1cebc1d55f8e0039aa239285e177",
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
        descriptive_stats={
            "n_samples": {"test": 6392},
            "avg_character_length": {
                "test": {
                    "average_document_length": 0.0,
                    "average_query_length": 0.0,
                    "num_documents": 6322,
                    "num_queries": 70,
                    "average_relevant_docs_per_query": 246.1,
                }
            },
        },
    )
    skip_first_result = False


class RParisHardI2IRetrieval(MultiChoiceEvaluationMixin, AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="RParisHardI2IRetrieval",
        description="Retrieve photos of landmarks in Paris, UK.",
        reference="https://openaccess.thecvf.com/content_cvpr_2018/html/Radenovic_Revisiting_Paris_and_CVPR_2018_paper.html",
        dataset={
            "path": "JamieSJS/r-paris-hard-multi",
            "revision": "4e5997e48fb2f2f8bf1c8973851dedeb17e09a83",
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
        descriptive_stats={
            "n_samples": {"test": 6392},
            "avg_character_length": {
                "test": {
                    "average_document_length": 0.0,
                    "average_query_length": 0.0,
                    "num_documents": 6322,
                    "num_queries": 70,
                    "average_relevant_docs_per_query": 147.86,
                }
            },
        },
    )
    skip_first_result = False
