from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class ThaiRestaurantRestaReviews(AbsTaskClassification):
    metadata = TaskMetadata(
        name="ThaiRestaurantReviews",
        description="Wongnai features over 200,000 restaurants, beauty salons, and spas across Thailand on its platform, with detailed information about each merchant and user reviews. In this dataset there are 5 classes corressponding each star rating",
        reference="https://github.com/wongnai/wongnai-corpus",
        dataset={
            "path": "wongnai_reviews",
            "revision": "e708d4545d7ab10dd2c6b5b5b2a72ca28685dae2",
        },
        type="Classification",
        category="p2p",
        eval_splits=["test"],
        eval_langs=["tha-Thai"],
        main_score="accuracy",
        date=("2018-01-01", "2018-12-31"),
        form=["written"],
        dialect=[],
        domains=["Reviews"],
        task_subtypes=[],
        license="LGPL-3.0",
        socioeconomic_status="mixed",
        annotations_creators="derived",
        text_creation="found",
        bibtex_citation="""
        @software{cstorm125_2020_3852912,
            author  = {cstorm125 and lukkiddd},
            title   = {PyThaiNLP/classification-benchmarks: v0.1-alpha},
            month   = may,
            year    = 2020,
            publisher = {Zenodo},
            version = {v0.1-alpha},
            doi     = {10.5281/zenodo.3852912},
            url     = {https://doi.org/10.5281/zenodo.3852912}
        }""",
        n_samples={"test": 2048},
        avg_character_length={"test": 540.3717},
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_columns(
            {"review_body": "text", "star_rating": "label"}
        )
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["test"]
        )
