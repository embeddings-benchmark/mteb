from __future__ import annotations

from mteb.abstasks.Image.AbsTaskAny2AnyRetrieval import AbsTaskAny2AnyRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


class Flickr30kI2TRetrieval(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="Flickr30kI2TRetrieval",
        description="Retrieve captions based on images.",
        reference="https://www.semanticscholar.org/paper/From-image-descriptions-to-visual-denotations%3A-New-Young-Lai/44040913380206991b1991daf1192942e038fe31",
        dataset={
            "path": "JamieSJS/flickr30k",
            "revision": "a4cf34ac79215f9e2cd6a10342d84f606fc41cc3",
        },
        type="Retrieval",
        category="i2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2018-01-01", "2018-12-31"),
        form=["written"],
        domains=["Web"],
        task_subtypes=["Image Text Retrieval"],
        license="CC BY-SA 4.0",
        socioeconomic_status="medium",
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
        descriptive_stats={
              "n_samples": {"default": 155070}, # qrels
        },
    )
