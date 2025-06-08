from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class CzechProductReviewSentimentClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="CzechProductReviewSentimentClassification",
        description="User reviews of products on Czech e-shop Mall.cz with 3 sentiment classes (positive, neutral, negative)",
        reference="https://aclanthology.org/W13-1609/",
        dataset={
            "path": "fewshot-goes-multilingual/cs_mall-product-reviews",
            "revision": "2e6fedf42c9c104e83dfd95c3a453721e683e244",
        },
        type="Classification",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["ces-Latn"],
        main_score="accuracy",
        date=("2013-01-01", "2013-06-01"),
        dialect=[],
        domains=["Reviews", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="cc-by-nc-sa-4.0",
        annotations_creators="derived",
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{habernal-etal-2013-sentiment,
  address = {Atlanta, Georgia},
  author = {Habernal, Ivan  and
Pt{\'a}{\v{c}}ek, Tom{\'a}{\v{s}}  and
Steinberger, Josef},
  booktitle = {Proceedings of the 4th Workshop on Computational Approaches to Subjectivity, Sentiment and Social Media Analysis},
  editor = {Balahur, Alexandra  and
van der Goot, Erik  and
Montoyo, Andres},
  month = jun,
  pages = {65--74},
  publisher = {Association for Computational Linguistics},
  title = {Sentiment Analysis in {C}zech Social Media Using Supervised Machine Learning},
  url = {https://aclanthology.org/W13-1609},
  year = {2013},
}
""",
    )
    samples_per_label = 16

    def dataset_transform(self) -> None:
        self.dataset = self.dataset.rename_columns(
            {"comment": "text", "rating_str": "label"}
        )
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["test"]
        )
