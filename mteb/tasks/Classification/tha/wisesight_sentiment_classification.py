from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class wisesight_sentiment_classification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="wisesight_sentiment",
        description="Wisesight Sentiment Corpus: Social media messages in Thai language with sentiment label (positive, neutral, negative, question)",
        reference="https://github.com/PyThaiNLP/wisesight-sentiment",
        dataset={
            "path": "wisesight_sentiment",
            "revision": "14aa5773afa135ba835cc5179bbc4a63657a42ae",
        },
        type="Classification",
        category="s2s",
        eval_splits=["train", "test", "validation"],
        eval_langs=["tha-Thai"],
        main_score="f1",
        date=("2019-05-24", "2021-09-16"),
        form=["written"],
        dialect=[],
        domains=["Social", "News"],
        task_subtypes=["Sentiment/Hate speech"],
        license="cc0-1.0",
        socioeconomic_status="mixed",
        annotations_creators="expert-annotated",
        text_creation="found",
        bibtex_citation="""@software{bact_2019_3457447,
  author       = {Suriyawongkul, Arthit and
                  Chuangsuwanich, Ekapol and
                  Chormai, Pattarawat and
                  Polpanumas, Charin},
  title        = {PyThaiNLP/wisesight-sentiment: First release},
  month        = sep,
  year         = 2019,
  publisher    = {Zenodo},
  version      = {v1.0},
  doi          = {10.5281/zenodo.3457447},
  url          = {https://doi.org/10.5281/zenodo.3457447}
}

""",
        n_samples={"train": 3418},
        avg_character_length={"train": 103.42},
    )

    def dataset_transform(self):
        N_SAMPLES = 2048
        TEST_SAMPLES = 1024  # define this as per your requirement

        # Transform train set
        self.dataset["train"] = self.dataset["train"].select(range(N_SAMPLES))
        self.dataset["train"] = self.dataset["train"].rename_column("texts", "text")
        self.dataset["train"] = self.dataset["train"].rename_column("category", "label")

        # Transform validation set
        self.dataset["validation"] = self.dataset["validation"].select(
            range(TEST_SAMPLES)
        )
        self.dataset["validation"] = self.dataset["validation"].rename_column(
            "texts", "text"
        )
        self.dataset["validation"] = self.dataset["validation"].rename_column(
            "category", "label"
        )

        # Transform test set
        self.dataset["test"] = self.dataset["test"].select(range(TEST_SAMPLES))
        self.dataset["test"] = self.dataset["test"].rename_column("texts", "text")
        self.dataset["test"] = self.dataset["test"].rename_column("category", "label")
