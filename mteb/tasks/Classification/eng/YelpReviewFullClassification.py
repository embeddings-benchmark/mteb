from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import AbsTaskClassification


class YelpReviewFullClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="YelpReviewFullClassification",
        description="Yelp Review Full is a dataset for sentiment analysis, containing 5 classes corresponding to ratings 1-5.",
        reference="https://arxiv.org/abs/1509.01626",
        dataset={
            "path": "yelp_review_full",
            "revision": "c1f9ee939b7d05667af864ee1cb066393154bf85",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2015-01-01", "2015-12-31"),  # reviews from 2015
        form=["written"],
        domains=["Reviews"],
        task_subtypes=None,
        license=None,
        socioeconomic_status="mixed",
        annotations_creators=None,
        dialect=None,
        text_creation=None,
        bibtex_citation=None,
        n_samples={"test": 50000},
        avg_character_length=None,
    )

    @property
    def metadata_dict(self) -> dict[str, str]:
        metadata_dict = dict(self.metadata)
        metadata_dict["n_experiments"] = 10
        metadata_dict["samples_per_label"] = 128
        return metadata_dict

    def dataset_transform(self):
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["test"]
        )
