from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class CzechSoMeSentimentClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="CzechSoMeSentimentClassification",
        description="User comments on Facebook",
        reference="https://aclanthology.org/W13-1609/",
        dataset={
            "path": "fewshot-goes-multilingual/cs_facebook-comments",
            "revision": "6ced1d87a030915822b087bf539e6d5c658f1988",
        },
        type="Classification",
        category="t2c",
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
        superseded_by="CzechSoMeSentimentClassification.v2",
    )
    samples_per_label = 16

    def dataset_transform(self) -> None:
        self.dataset = self.dataset.rename_columns(
            {"comment": "text", "sentiment_int": "label"}
        )


class CzechSoMeSentimentClassificationV2(AbsTaskClassification):
    metadata = TaskMetadata(
        name="CzechSoMeSentimentClassification.v2",
        description="User comments on Facebook This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)",
        reference="https://aclanthology.org/W13-1609/",
        dataset={
            "path": "mteb/czech_so_me_sentiment",
            "revision": "a12152e40ff9857bf3c83694528f40ec5c02aafc",
        },
        type="Classification",
        category="t2c",
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
        adapted_from=["CzechSoMeSentimentClassification"],
    )
    samples_per_label = 16
