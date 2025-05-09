from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class DanishPoliticalCommentsClassification(AbsTaskClassification):
    superseded_by = "DanishPoliticalCommentsClassification.v2"
    metadata = TaskMetadata(
        name="DanishPoliticalCommentsClassification",
        dataset={
            "path": "community-datasets/danish_political_comments",
            "revision": "edbb03726c04a0efab14fc8c3b8b79e4d420e5a1",
            "trust_remote_code": True,
        },
        description="A dataset of Danish political comments rated for sentiment",
        reference="https://huggingface.co/datasets/danish_political_comments",
        type="Classification",
        category="s2s",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs=["dan-Latn"],
        main_score="accuracy",
        date=(
            "2000-01-01",
            "2022-12-31",
        ),  # Estimated range for the collection of comments
        domains=["Social", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@techreport{SAMsentiment,
  author = {Mads Guldborg Kjeldgaard Kongsbak and Steffan Eybye Christensen and Lucas Høyberg Puvis~de~Chavannes and Peter Due Jensen},
  institution = {IT University of Copenhagen},
  title = {Sentiment Analysis Multitool, SAM},
  year = {2019},
}
""",
        prompt="Classify Danish political comments for sentiment",
    )

    samples_per_label = 16

    def dataset_transform(self):
        self.dataset = self.dataset.rename_column("sentence", "text")
        self.dataset = self.dataset.rename_column("target", "label")

        # create train and test splits
        self.dataset = self.dataset["train"].train_test_split(0.2, seed=self.seed)


class DanishPoliticalCommentsClassificationV2(AbsTaskClassification):
    metadata = TaskMetadata(
        name="DanishPoliticalCommentsClassification.v2",
        dataset={
            "path": "mteb/danish_political_comments",
            "revision": "99f9768194180731a5199f384b10a897ece58687",
        },
        description="""A dataset of Danish political comments rated for sentiment
        This version corrects errors found in the original data. For details, see https://github.com/embeddings-benchmark/mteb/pull/2680""",
        reference="https://huggingface.co/datasets/danish_political_comments",
        type="Classification",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["dan-Latn"],
        main_score="accuracy",
        date=(
            "2000-01-01",
            "2022-12-31",
        ),  # Estimated range for the collection of comments
        domains=["Social", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@techreport{SAMsentiment,
  author = {Mads Guldborg Kjeldgaard Kongsbak and Steffan Eybye Christensen and Lucas Høyberg Puvis~de~Chavannes and Peter Due Jensen},
  institution = {IT University of Copenhagen},
  title = {Sentiment Analysis Multitool, SAM},
  year = {2019},
}
""",
        prompt="Classify Danish political comments for sentiment",
    )

    samples_per_label = 16
