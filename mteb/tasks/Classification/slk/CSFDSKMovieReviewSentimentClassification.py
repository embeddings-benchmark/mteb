from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata

N_SAMPLES = 2048


class CSFDSKMovieReviewSentimentClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="CSFDSKMovieReviewSentimentClassification",
        description="The dataset contains 30k user reviews from csfd.cz in Slovak.",
        reference="https://arxiv.org/abs/2304.01922",
        dataset={
            "path": "fewshot-goes-multilingual/sk_csfd-movie-reviews",
            "revision": "23a20c659d868740ef9c54854de631fe19cd5c17",
        },
        type="Classification",
        category="s2s",
        modalities=["text"],
        date=("2002-05-21", "2020-03-05"),
        eval_splits=["test"],
        eval_langs=["slk-Latn"],
        main_score="accuracy",
        domains=["Reviews", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""
@misc{štefánik2023resources,
      title={Resources and Few-shot Learners for In-context Learning in Slavic Languages}, 
      author={Michal Štefánik and Marek Kadlčík and Piotr Gramacki and Petr Sojka},
      year={2023},
      eprint={2304.01922},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
""",
        descriptive_stats={
            "n_samples": {"test": N_SAMPLES},
            "avg_character_length": {"test": 366.2},
        },
    )

    @property
    def metadata_dict(self):
        md = super().metadata_dict
        # Increase the samples_per_label in order to improve baseline performance
        md["samples_per_label"] = 20
        return md

    def dataset_transform(self):
        self.dataset = self.dataset.rename_columns(
            {"comment": "text", "rating_int": "label"}
        )
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["test"], n_samples=N_SAMPLES
        )
