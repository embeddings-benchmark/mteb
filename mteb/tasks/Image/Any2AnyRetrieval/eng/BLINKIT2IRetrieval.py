from __future__ import annotations

from mteb.abstasks.Image.AbsTaskAny2AnyRetrieval import AbsTaskAny2AnyRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


class BLINKIT2IRetrieval(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="BLINKIT2IRetrieval",
        description="Retrieve images based on images and specific retrieval instructions.",
        reference="https://arxiv.org/abs/2404.12390",
        dataset={
            "path": "JamieSJS/blink-it2i",
            "revision": "7a1a1330565faca9c1aeec6f5acfc64f21296753",
            "trust_remote_code": True,
        },
        type="Any2AnyRetrieval",
        category="it2i",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2018-01-01", "2018-12-31"),
        domains=["Encyclopaedic"],
        task_subtypes=["Image Text Retrieval"],
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation="""@article{fu2024blink,
  title={Blink: Multimodal large language models can see but not perceive},
  author={Fu, Xingyu and Hu, Yushi and Li, Bangzheng and Feng, Yu and Wang, Haoyu and Lin, Xudong and Roth, Dan and Smith, Noah A and Ma, Wei-Chiu and Krishna, Ranjay},
  journal={arXiv preprint arXiv:2404.12390},
  year={2024}
}
""",
        descriptive_stats={
            "n_samples": {"test": 402},
            "avg_character_length": {
                "test": {
                    "average_document_length": 0.0,
                    "average_query_length": 0.0,
                    "num_documents": 804,
                    "num_queries": 402,
                    "average_relevant_docs_per_query": 1,
                }
            },
        },
    )
