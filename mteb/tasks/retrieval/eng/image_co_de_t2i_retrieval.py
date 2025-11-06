from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class ImageCoDeT2IRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="ImageCoDeT2IRetrieval",
        description="Retrieve a specific video frame based on a precise caption.",
        reference="https://aclanthology.org/2022.acl-long.241.pdf",
        dataset={
            "path": "JamieSJS/imagecode",
            "revision": "a424cd523ffb157b69a875fb5e71c1d51be54089",
        },
        type="Any2AnyRetrieval",
        category="t2i",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="cv_recall_at_3",
        date=("2022-05-22", "2022-05-27"),  # conference dates
        domains=["Web", "Written"],
        task_subtypes=["Image Text Retrieval"],
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation=r"""
@article{krojer2022image,
  author = {Krojer, Benno and Adlakha, Vaibhav and Vineet, Vibhav and Goyal, Yash and Ponti, Edoardo and Reddy, Siva},
  journal = {arXiv preprint arXiv:2203.15867},
  title = {Image retrieval from contextual descriptions},
  year = {2022},
}
""",
    )
