from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class AmazonReviewsClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="AmazonReviewsClassification",
        dataset={
            "path": "mteb/AmazonReviewsClassification",
            "revision": "6b5d328eaae8ef408dd7d775040245cf86f92e9d",
        },
        description="A collection of Amazon reviews specifically designed to aid research in multilingual text classification.",
        reference="https://arxiv.org/abs/2010.02573",
        category="t2c",
        modalities=["text"],
        type="Classification",
        eval_splits=["validation", "test"],
        eval_langs={
            "en": ["eng-Latn"],
            "de": ["deu-Latn"],
            "es": ["spa-Latn"],
            "fr": ["fra-Latn"],
            "ja": ["jpn-Jpan"],
            "zh": ["cmn-Hans"],
        },
        main_score="accuracy",
        date=("2015-11-01", "2019-11-01"),
        domains=["Reviews", "Written"],
        task_subtypes=[],
        license="https://docs.opendata.aws/amazon-reviews-ml/license.txt",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@misc{keung2020multilingual,
  archiveprefix = {arXiv},
  author = {Phillip Keung and Yichao Lu and György Szarvas and Noah A. Smith},
  eprint = {2010.02573},
  primaryclass = {cs.CL},
  title = {The Multilingual Amazon Reviews Corpus},
  year = {2020},
}
""",
        prompt="Classify the given Amazon review into its appropriate rating category",
    )
