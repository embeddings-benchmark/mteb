from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.MultilingualTask import MultilingualTask
from mteb.abstasks.TaskMetadata import TaskMetadata


class AmazonReviewsClassification(MultilingualTask, AbsTaskClassification):
    metadata = TaskMetadata(
        name="AmazonReviewsClassification",
        dataset={
            "path": "mteb/amazon_reviews_multi",
            "revision": "1399c76144fd37290681b995c656ef9b2e06e26d",
            "trust_remote_code": True,
        },
        description="A collection of Amazon reviews specifically designed to aid research in multilingual text classification.",
        reference="https://arxiv.org/abs/2010.02573",
        category="s2s",
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
