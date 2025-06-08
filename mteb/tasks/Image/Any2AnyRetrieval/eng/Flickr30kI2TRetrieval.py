from __future__ import annotations

from mteb.abstasks.Image.AbsTaskAny2AnyRetrieval import AbsTaskAny2AnyRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


class Flickr30kI2TRetrieval(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="Flickr30kI2TRetrieval",
        description="Retrieve captions based on images.",
        reference="https://www.semanticscholar.org/paper/From-image-descriptions-to-visual-denotations%3A-New-Young-Lai/44040913380206991b1991daf1192942e038fe31",
        dataset={
            "path": "isaacchung/flickr30ki2t",
            "revision": "6984df6bd4380034e7766d9a992d8907df363efb",
        },
        type="Any2AnyRetrieval",
        category="i2t",
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
        bibtex_citation=r"""
@article{Young2014FromID,
  author = {Peter Young and Alice Lai and Micah Hodosh and J. Hockenmaier},
  journal = {Transactions of the Association for Computational Linguistics},
  pages = {67-78},
  title = {From image descriptions to visual denotations: New similarity metrics for semantic inference over event descriptions},
  url = {https://api.semanticscholar.org/CorpusID:3104920},
  volume = {2},
  year = {2014},
}
""",
        prompt={"query": "Find an image caption describing the following image."},
        descriptive_stats={
            "n_samples": {"test": 1000},
            "avg_character_length": {
                "test": {
                    "average_document_length": 0.0,
                    "average_query_length": 0.0,
                    "num_documents": 5000,
                    "num_queries": 1000,
                }
            },
        },
    )
