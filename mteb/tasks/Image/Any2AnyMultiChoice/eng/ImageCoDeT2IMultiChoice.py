from __future__ import annotations

from mteb.abstasks.Image.AbsTaskAny2AnyMultiChoice import AbsTaskAny2AnyMultiChoice
from mteb.abstasks.TaskMetadata import TaskMetadata


class ImageCoDeT2IMultiChoice(AbsTaskAny2AnyMultiChoice):
    metadata = TaskMetadata(
        name="ImageCoDeT2IMultiChoice",
        description="Identify the correct image from a set of similar images based on a precise caption.",
        reference="https://aclanthology.org/2022.acl-long.241.pdf",
        dataset={
            "path": "JamieSJS/imagecode-multi",
            "revision": "d28adfd8b34fefa546fdf94bdc352622b2575f6c",
        },
        type="Compositionality",
        category="it2i",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2022-05-22", "2022-05-27"),  # conference dates
        domains=["Web", "Written"],
        task_subtypes=["Image Text Retrieval"],
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation="""@article{krojer2022image,
  title={Image retrieval from contextual descriptions},
  author={Krojer, Benno and Adlakha, Vaibhav and Vineet, Vibhav and Goyal, Yash and Ponti, Edoardo and Reddy, Siva},
  journal={arXiv preprint arXiv:2203.15867},
  year={2022}
}
""",
        descriptive_stats={
            "n_samples": {"test": 2302},
            "avg_character_length": {
                "test": {
                    "average_document_length": 0.0,
                    "average_query_length": 0.0,
                    "num_documents": 23020,
                    "num_queries": 2302,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
    )
