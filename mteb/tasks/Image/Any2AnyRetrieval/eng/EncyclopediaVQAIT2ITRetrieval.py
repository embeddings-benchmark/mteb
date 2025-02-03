from __future__ import annotations

from mteb.abstasks.Image.AbsTaskAny2AnyRetrieval import AbsTaskAny2AnyRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


class EncyclopediaVQAIT2ITRetrieval(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="EncyclopediaVQAIT2ITRetrieval",
        description="Retrieval Wiki passage and image and passage to answer query about an image.",
        reference="https://github.com/google-research/google-research/tree/master/encyclopedic_vqa",
        dataset={
            "path": "izhx/UMRB-EncyclopediaVQA",
            "revision": "d6eae4f06e260664eb3f276fd1bdb5d4d4c9f32b",
        },
        type="Any2AnyRetrieval",
        category="it2it",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="cv_recall_at_5",
        date=("2023-01-01", "2023-07-20"),
        domains=["Encyclopaedic"],
        task_subtypes=["Image Text Retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        modalities=["image", "text"],
        sample_creation="created",
        bibtex_citation="""@inproceedings{mensink2023encyclopedic,
  title={Encyclopedic VQA: Visual questions about detailed properties of fine-grained categories},
  author={Mensink, Thomas and Uijlings, Jasper and Castrejon, Lluis and Goel, Arushi and Cadar, Felipe and Zhou, Howard and Sha, Fei and Araujo, Andr{\'e} and Ferrari, Vittorio},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={3113--3124},
  year={2023}
}""",
        prompt={
            "query": "Obtain illustrated documents that correspond to the inquiry alongside the provided image."
        },
        descriptive_stats={
            "n_samples": {"test": 3743},
            "avg_character_length": {
                "test": {
                    "average_document_length": 1294.368802424136,
                    "average_query_length": 51.703713598717606,
                    "num_documents": 68313,
                    "num_queries": 3743,
                    "average_relevant_docs_per_query": 1.3056371894202512,
                }
            },
        },
    )
