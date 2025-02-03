from __future__ import annotations

from mteb.abstasks.Image.AbsTaskAny2AnyRetrieval import AbsTaskAny2AnyRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


class Flickr30kT2IRetrieval(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="Flickr30kT2IRetrieval",
        description="Retrieve images based on captions.",
        reference="https://www.semanticscholar.org/paper/From-image-descriptions-to-visual-denotations%3A-New-Young-Lai/44040913380206991b1991daf1192942e038fe31",
        dataset={
            "path": "isaacchung/flickr30kt2i",
            "revision": "e819702b287bfbe084e129a61f308a802b7c108e",
        },
        type="Any2AnyRetrieval",
        category="t2i",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2018-01-01", "2018-12-31"),
        domains=["Web", "Written"],
        task_subtypes=["Image Text Retrieval"],
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation="""@article{Young2014FromID,
  title={From image descriptions to visual denotations: New similarity metrics for semantic inference over event descriptions},
  author={Peter Young and Alice Lai and Micah Hodosh and J. Hockenmaier},
  journal={Transactions of the Association for Computational Linguistics},
  year={2014},
  volume={2},
  pages={67-78},
  url={https://api.semanticscholar.org/CorpusID:3104920}
}""",
        prompt={"query": "Find an image that matches the given caption."},
        descriptive_stats={
            "n_samples": {"test": 5000},
            "avg_character_length": {
                "test": {
                    "average_document_length": 0.0,
                    "average_query_length": 0.0,
                    "num_documents": 1000,
                    "num_queries": 5000,
                }
            },
        },
    )
