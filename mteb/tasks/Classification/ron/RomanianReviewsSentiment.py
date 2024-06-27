from __future__ import annotations

from mteb.abstasks import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata

N_SAMPLES = 2048


class RomanianReviewsSentiment(AbsTaskClassification):
    metadata = TaskMetadata(
        name="RomanianReviewsSentiment",
        description="LaRoSeDa (A Large Romanian Sentiment Data Set) contains 15,000 reviews written in Romanian",
        reference="https://arxiv.org/abs/2101.04197",
        dataset={
            "path": "universityofbucharest/laroseda",
            "revision": "358bcc95aeddd5d07a4524ee416f03d993099b23",
            "trust_remote_code": True,
        },
        type="Classification",
        category="s2s",
        date=("2020-01-01", "2021-01-11"),
        eval_splits=["test"],
        eval_langs=["ron-Latn"],
        main_score="accuracy",
        form=["written"],
        domains=["Reviews"],
        task_subtypes=["Sentiment/Hate speech"],
        license="CC-BY-4.0",
        socioeconomic_status="mixed",
        annotations_creators="derived",
        dialect=[],
        text_creation="found",
        bibtex_citation="""
@article{
    tache2101clustering,
    title={Clustering Word Embeddings with Self-Organizing Maps. Application on LaRoSeDa -- A Large Romanian Sentiment Data Set},
    author={Anca Maria Tache and Mihaela Gaman and Radu Tudor Ionescu},
    journal={ArXiv},
    year = {2021}
}
""",
        n_samples={"test": N_SAMPLES},
        avg_character_length={"test": 588.6},
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_columns(
            {"content": "text", "starRating": "label"}
        )
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["test"]
        )
