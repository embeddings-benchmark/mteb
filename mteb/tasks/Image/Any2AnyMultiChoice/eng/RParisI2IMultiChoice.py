from __future__ import annotations

from mteb.abstasks.Image.AbsTaskAny2AnyMultiChoice import AbsTaskAny2AnyMultiChoice
from mteb.abstasks.TaskMetadata import TaskMetadata


# NOTE: These tasks are marked as Any2AnyRetrieval types they are the correct implementations of ROxford retrieval and RParis retrieval
# (as it requires masking out the different docs in corpus for every query). This aligns with the MIEB papeer.
class RParisEasyI2IMultiChoice(AbsTaskAny2AnyMultiChoice):
    metadata = TaskMetadata(
        name="RParisEasyI2IMultiChoice",
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
        main_score="accuracy",
        date=("2009-01-01", "2010-04-01"),
        domains=["Web"],
        task_subtypes=["Object recognition"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        modalities=["image"],
        sample_creation="created",
        bibtex_citation="""@inproceedings{radenovic2018revisiting,
  title={Revisiting paris and paris: Large-scale image MultiChoice benchmarking},
  author={Radenovi{\'c}, Filip and Iscen, Ahmet and Tolias, Giorgos and Avrithis, Yannis and Chum, Ond{\v{r}}ej},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={5706--5715},
  year={2018}
}
        """,
        descriptive_stats={
            "n_samples": {"test": 70},
            "avg_character_length": {
                "test": {
                    "average_document_length": 0.0,
                    "average_query_length": 0.0,
                    "num_documents": 516,
                    "num_queries": 70,
                    "average_relevant_docs_per_query": 43.3,
                }
            },
        },
    )
    skip_first_result = False


class RParisMediumI2IMultiChoice(AbsTaskAny2AnyMultiChoice):
    metadata = TaskMetadata(
        name="RParisMediumI2IMultiChoice",
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
        main_score="accuracy",
        date=("2009-01-01", "2010-04-01"),
        domains=["Web"],
        task_subtypes=["Object recognition"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        modalities=["image"],
        sample_creation="created",
        bibtex_citation="""@inproceedings{radenovic2018revisiting,
  title={Revisiting paris and paris: Large-scale image MultiChoice benchmarking},
  author={Radenovi{\'c}, Filip and Iscen, Ahmet and Tolias, Giorgos and Avrithis, Yannis and Chum, Ond{\v{r}}ej},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={5706--5715},
  year={2018}
}
        """,
        descriptive_stats={
            "n_samples": {"test": 70},
            "avg_character_length": {
                "test": {
                    "average_document_length": 0.0,
                    "average_query_length": 0.0,
                    "num_documents": 788,
                    "num_queries": 70,
                    "average_relevant_docs_per_query": 78.9,
                }
            },
        },
    )
    skip_first_result = False


class RParisHardI2IMultiChoice(AbsTaskAny2AnyMultiChoice):
    metadata = TaskMetadata(
        name="RParisHardI2IMultiChoice",
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
        main_score="accuracy",
        date=("2009-01-01", "2010-04-01"),
        domains=["Web"],
        task_subtypes=["Object recognition"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        modalities=["image"],
        sample_creation="created",
        bibtex_citation="""@inproceedings{radenovic2018revisiting,
  title={Revisiting paris and paris: Large-scale image MultiChoice benchmarking},
  author={Radenovi{\'c}, Filip and Iscen, Ahmet and Tolias, Giorgos and Avrithis, Yannis and Chum, Ond{\v{r}}ej},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={5706--5715},
  year={2018}
}
        """,
        descriptive_stats={
            "n_samples": {"test": 70},
            "avg_character_length": {
                "test": {
                    "average_document_length": 0.0,
                    "average_query_length": 0.0,
                    "num_documents": 685,
                    "num_queries": 70,
                    "average_relevant_docs_per_query": 35.7,
                }
            },
        },
    )
    skip_first_result = False
