from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class EncyclopediaVQAIT2ITRetrieval(AbsTaskRetrieval):
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
        bibtex_citation=r"""
@inproceedings{mensink2023encyclopedic,
  author = {Mensink, Thomas and Uijlings, Jasper and Castrejon, Lluis and Goel, Arushi and Cadar, Felipe and Zhou, Howard and Sha, Fei and Araujo, Andr{\'e} and Ferrari, Vittorio},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages = {3113--3124},
  title = {Encyclopedic VQA: Visual questions about detailed properties of fine-grained categories},
  year = {2023},
}
""",
        prompt={
            "query": "Obtain illustrated documents that correspond to the inquiry alongside the provided image."
        },
    )
