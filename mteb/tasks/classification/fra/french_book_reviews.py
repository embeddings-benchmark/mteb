from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class FrenchBookReviews(AbsTaskClassification):
    metadata = TaskMetadata(
        name="FrenchBookReviews",
        dataset={
            "path": "Abirate/french_book_reviews",
            "revision": "534725e03fec6f560dbe8166e8ae3825314a6290",
        },
        description="It is a French book reviews dataset containing a huge number of reader reviews on French books. Each review is pared with a rating that ranges from 0.5 to 5 (with 0.5 increment).",
        reference="https://huggingface.co/datasets/Abirate/french_book_reviews",
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs=["fra-Latn"],
        main_score="accuracy",
        date=("2022-01-01", "2023-01-01"),
        domains=["Reviews", "Written"],
        task_subtypes=[],
        license="cc0-1.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""
        """,
        superseded_by="FrenchBookReviews.v2",
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_columns({"reader_review": "text"})
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["train"]
        )


class FrenchBookReviewsV2(AbsTaskClassification):
    metadata = TaskMetadata(
        name="FrenchBookReviews.v2",
        dataset={
            "path": "mteb/french_book_reviews",
            "revision": "71d755fd76073533c3d0c262f6b542eb0fa7ce96",
        },
        description="It is a French book reviews dataset containing a huge number of reader reviews on French books. Each review is pared with a rating that ranges from 0.5 to 5 (with 0.5 increment). This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)",
        reference="https://huggingface.co/datasets/Abirate/french_book_reviews",
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fra-Latn"],
        main_score="accuracy",
        date=("2022-01-01", "2023-01-01"),
        domains=["Reviews", "Written"],
        task_subtypes=[],
        license="cc0-1.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""
        """,
        adapted_from=["FrenchBookReviews"],
    )

    def dataset_transform(self):
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["train"]
        )
