from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import AbsTaskClassification, MultilingualTask


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
        bibtex_citation="""@misc{keung2020multilingual,
      title={The Multilingual Amazon Reviews Corpus}, 
      author={Phillip Keung and Yichao Lu and György Szarvas and Noah A. Smith},
      year={2020},
      eprint={2010.02573},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}""",
        descriptive_stats={
            "n_samples": {"validation": 30000, "test": 30000},
            "avg_character_length": {"validation": 159.2, "test": 160.4},
        },
    )
