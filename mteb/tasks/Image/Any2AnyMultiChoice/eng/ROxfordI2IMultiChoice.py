from __future__ import annotations

from mteb.abstasks.Image.AbsTaskAny2AnyMultiChoice import AbsTaskAny2AnyMultiChoice
from mteb.abstasks.TaskMetadata import TaskMetadata


class ROxfordEasyI2IMultiChoice(AbsTaskAny2AnyMultiChoice):
    metadata = TaskMetadata(
        name="ROxfordEasyI2IMultiChoice",
        description="Retrieve photos of landmarks in Oxford, UK.",
        reference="https://openaccess.thecvf.com/content_cvpr_2018/html/Radenovic_Revisiting_Oxford_and_CVPR_2018_paper.html",
        dataset={
            "path": "JamieSJS/r-oxford-easy-multi",
            "revision": "8fde638cd1a297b81f7e26914a39b58788a3e4b4",
        },
        type="Any2AnyMultiChoice",
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
  title={Revisiting oxford and paris: Large-scale image MultiChoice benchmarking},
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


class ROxfordMediumI2IMultiChoice(AbsTaskAny2AnyMultiChoice):
    metadata = TaskMetadata(
        name="ROxfordMediumI2IMultiChoice",
        description="Retrieve photos of landmarks in Oxford, UK.",
        reference="https://openaccess.thecvf.com/content_cvpr_2018/html/Radenovic_Revisiting_Oxford_and_CVPR_2018_paper.html",
        dataset={
            "path": "JamieSJS/r-oxford-medium-multi",
            "revision": "3aed4b566b6ea51b81ee50e6cbf18db8349893c0",
        },
        type="Any2AnyMultiChoice",
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
  title={Revisiting oxford and paris: Large-scale image MultiChoice benchmarking},
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


class ROxfordHardI2IMultiChoice(AbsTaskAny2AnyMultiChoice):
    metadata = TaskMetadata(
        name="ROxfordHardI2IMultiChoice",
        description="Retrieve photos of landmarks in Oxford, UK.",
        reference="https://openaccess.thecvf.com/content_cvpr_2018/html/Radenovic_Revisiting_Oxford_and_CVPR_2018_paper.html",
        dataset={
            "path": "JamieSJS/r-oxford-hard-multi",
            "revision": "93d282cc623d122d583b6db3837f07d91a7b5575",
        },
        type="Any2AnyMultiChoice",
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
  title={Revisiting oxford and paris: Large-scale image MultiChoice benchmarking},
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
