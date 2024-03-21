from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import AbsTaskClassification, MultilingualTask

_LANGUAGES = ["en", "de", "es", "fr", "ja", "zh"]


class AmazonReviewsClassification(MultilingualTask, AbsTaskClassification):
    metadata = TaskMetadata(
        name="AmazonReviewsClassification",
        hf_hub_name="mteb/amazon_reviews_multi",
        description="A collection of Amazon reviews specifically designed to aid research in multilingual text classification.",
        reference="https://arxiv.org/abs/2010.02573",
        category="s2s",
        type="Classification",
        eval_splits=["validation", "test"],
        eval_langs=_LANGUAGES,
        main_score="accuracy",
        revision="1399c76144fd37290681b995c656ef9b2e06e26d",
        date=None,
        form=None,
        domains=None,
        task_subtypes=None,
        license=None,
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=None,
        text_creation=None,
        bibtex_citation=None,
        n_samples={"validation": 30000, "test": 30000},
        avg_character_length={"validation": 159.2, "test": 160.4},
    )
