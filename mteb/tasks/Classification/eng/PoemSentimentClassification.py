from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class PoemSentimentClassification(AbsTaskClassification):
    superseded_by = "PoemSentimentClassification.v2"
    metadata = TaskMetadata(
        name="PoemSentimentClassification",
        description="Poem Sentiment is a sentiment dataset of poem verses from Project Gutenberg.",
        reference="https://arxiv.org/abs/2011.02686",
        dataset={
            "path": "google-research-datasets/poem_sentiment",
            "revision": "329d529d875a00c47ec71954a1a96ae167584770",
            "trust_remote_code": True,
        },
        type="Classification",
        category="s2s",
        modalities=["text"],
        eval_splits=["validation", "test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("1700-01-01", "1900-01-01"),
        domains=["Reviews", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=["eng-Latn-US", "en-Latn-GB"],
        sample_creation="found",
        bibtex_citation=r"""
@misc{sheng2020investigating,
  archiveprefix = {arXiv},
  author = {Emily Sheng and David Uthus},
  eprint = {2011.02686},
  primaryclass = {cs.CL},
  title = {Investigating Societal Biases in a Poetry Composition System},
  year = {2020},
}
""",
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_column("verse_text", "text")


class PoemSentimentClassificationV2(AbsTaskClassification):
    metadata = TaskMetadata(
        name="PoemSentimentClassification.v2",
        description="""Poem Sentiment is a sentiment dataset of poem verses from Project Gutenberg.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)""",
        reference="https://arxiv.org/abs/2011.02686",
        dataset={
            "path": "mteb/poem_sentiment",
            "revision": "9fdc57b89ccc09a8d9256f376112d626878e51a7",
        },
        type="Classification",
        category="s2s",
        modalities=["text"],
        eval_splits=["validation", "test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("1700-01-01", "1900-01-01"),
        domains=["Reviews", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=["eng-Latn-US", "en-Latn-GB"],
        sample_creation="found",
        bibtex_citation=r"""
@misc{sheng2020investigating,
  archiveprefix = {arXiv},
  author = {Emily Sheng and David Uthus},
  eprint = {2011.02686},
  primaryclass = {cs.CL},
  title = {Investigating Societal Biases in a Poetry Composition System},
  year = {2020},
}
""",
        adapted_from=["PoemSentimentClassification"],
    )
