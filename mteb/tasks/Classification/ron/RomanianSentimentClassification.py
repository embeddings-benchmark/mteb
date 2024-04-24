from __future__ import annotations

from mteb.abstasks import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata

TEST_SAMPLES = 2048


class RomanianSentimentClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="RomanianSentimentClassification",
        description="An Romanian dataset for sentiment classification.",
        reference="https://arxiv.org/abs/2009.08712",
        dataset={
            "path": "ro_sent",
            "revision": "155048684cea7a6d6af1ddbfeb9a04820311ce93",
        },
        type="Classification",
        category="s2s",
        date=("2020-09-18", "2020-09-18"),
        eval_splits=["test"],
        eval_langs=["ron-Latn"],
        main_score="accuracy",
        form=["written"],
        domains=["Reviews"],
        task_subtypes=["Sentiment/Hate speech"],
        license="Not specified",
        socioeconomic_status="mixed",
        annotations_creators="human-annotated",
        dialect=[],
        text_creation="found",
        bibtex_citation="""@article{dumitrescu2020birth,
  title={The birth of Romanian BERT},
  author={Dumitrescu, Stefan Daniel and Avram, Andrei-Marius and Pyysalo, Sampo},
  journal={arXiv preprint arXiv:2009.08712},
  year={2020}
}
""",
        n_samples={"test": TEST_SAMPLES},
        avg_character_length={"test": 67.6},
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_column("sentence", "text")
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["test"]
        )
