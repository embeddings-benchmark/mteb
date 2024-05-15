from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class SlovakMovieReviewSentimentClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SlovakMovieReviewSentimentClassification",
        description="User reviews of movies on the CSFD movie database, with 2 sentiment classes (positive, negative)",
        reference="https://arxiv.org/pdf/2304.01922",
        dataset={
            "path": "janko/sk_csfd-movie-reviews",
            "revision": "0c47583c9d339b3b6f89e4db76088af5f1ec8d39",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["svk-Latn"],
        main_score="accuracy",
        date=("2002-05-21", "2020-03-05"),
        form=["written"],
        dialect=[],
        domains=["Reviews"],
        task_subtypes=["Sentiment/Hate speech"],
        license="CC BY-NC-SA 4.0",
        socioeconomic_status="mixed",
        annotations_creators="derived",
        text_creation="found",
        bibtex_citation="""
        @article{vstefanik2023resources,
            title={Resources and Few-shot Learners for In-context Learning in Slavic Languages},
            author={{\v{S}}tef{\'a}nik, Michal and Kadl{\v{c}}{\'\i}k, Marek and Gramacki, Piotr and Sojka, Petr},
            journal={arXiv preprint arXiv:2304.01922},
            year={2023}
            }
        """,
        n_samples={"test": 2048},
        avg_character_length={"test": 366.17},
    )

    def dataset_transform(self) -> None:
        self.dataset = self.dataset.rename_columns({"comment": "text"})

        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["test"]
        )
