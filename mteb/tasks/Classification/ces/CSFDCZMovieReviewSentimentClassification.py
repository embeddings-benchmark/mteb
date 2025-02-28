from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class CSFDCZMovieReviewSentimentClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="CSFDCZMovieReviewSentimentClassification",
        description="The dataset contains 30k user reviews from csfd.cz in Czech.",
        reference="https://arxiv.org/abs/2304.01922",
        dataset={
            "path": "fewshot-goes-multilingual/cs_csfd-movie-reviews",
            "revision": "dd2ede6faaea338ef6b1e2966f06808656975a23",
        },
        type="Classification",
        category="s2s",
        modalities=["text"],
        date=("2002-06-28", "2020-03-13"),
        eval_splits=["test"],
        eval_langs=["ces-Latn"],
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
    )
    # Increase the samples_per_label in order to improve baseline performance
    samples_per_label = 20

    def dataset_transform(self):
        N_SAMPLES = 2048

        self.dataset = self.dataset.rename_columns(
            {"comment": "text", "rating_int": "label"}
        )
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["test"], n_samples=N_SAMPLES
        )
