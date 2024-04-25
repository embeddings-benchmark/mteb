from __future__ import annotations

from mteb.abstasks import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class SentLenClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SentLenClassification",
        description="This is a classification task where the goal is to predict the sentence length which has been binned in 6 possible categories with lengths ranging in the following intervals: --0: (5-8), 1: (9-12), 2: (13-16), 3: (17-20), 4: (21-25), 5: (26-28).",
        reference="https://arxiv.org/abs/1805.01070",
        dataset={
            "path": "tasksource/linguisticprobing",
            "revision": "1372f0696870b8ad09ec3ed123173be722c7a891",
            "name": "sentence_length",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2018-01-01", "2018-07-08"),
        form=["written"],
        domains=["Non-fiction", "Fiction"],
        task_subtypes=["Topic classification"],
        license="BSD",
        socioeconomic_status="mixed",
        annotations_creators="expert-annotated",
        dialect=[],
        text_creation="found",
        bibtex_citation="""
        @misc{conneau2018cram,
      title={What you can cram into a single vector: Probing sentence embeddings for linguistic properties}, 
      author={Alexis Conneau and German Kruszewski and Guillaume Lample and Loïc Barrault and Marco Baroni},
      year={2018},
      eprint={1805.01070},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
        }
        """,
        n_samples={"test": 8507},
        avg_character_length={"test": 86.80},
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_column("sentence", "text")
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["test"]
        )


class WCClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="WCClassification",
        description="This is a classification task with 1000 words as targets. The task is predicting which of the target words appear on the given sentence.",
        reference="https://arxiv.org/abs/1805.01070",
        dataset={
            "path": "tasksource/linguisticprobing",
            "revision": "1372f0696870b8ad09ec3ed123173be722c7a891",
            "name": "word_content",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2018-01-01", "2018-07-08"),
        form=["written"],
        domains=["Non-fiction", "Fiction"],
        task_subtypes=["Topic classification"],
        license="BSD",
        socioeconomic_status="mixed",
        annotations_creators="expert-annotated",
        dialect=[],
        text_creation="found",
        bibtex_citation="""
        @misc{conneau2018cram,
      title={What you can cram into a single vector: Probing sentence embeddings for linguistic properties}, 
      author={Alexis Conneau and German Kruszewski and Guillaume Lample and Loïc Barrault and Marco Baroni},
      year={2018},
      eprint={1805.01070},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
        }
        """,
        n_samples={"test": 8458},
        avg_character_length={"test": 82.67},
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_column("sentence", "text")
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["test"]
        )


class TreeDepthClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="TreeDepthClassification",
        description="This is a classification tasks where the goal is to predict the maximum depth of the sentence's syntactic tree (with values ranging from 5 to 12).",
        reference="https://arxiv.org/abs/1805.01070",
        dataset={
            "path": "tasksource/linguisticprobing",
            "revision": "1372f0696870b8ad09ec3ed123173be722c7a891",
            "name": "tree_depth",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2018-01-01", "2018-07-08"),
        form=["written"],
        domains=["Non-fiction", "Fiction"],
        task_subtypes=["Topic classification"],
        license="BSD",
        socioeconomic_status="mixed",
        annotations_creators="expert-annotated",
        dialect=[],
        text_creation="found",
        bibtex_citation="""
        @misc{conneau2018cram,
      title={What you can cram into a single vector: Probing sentence embeddings for linguistic properties}, 
      author={Alexis Conneau and German Kruszewski and Guillaume Lample and Loïc Barrault and Marco Baroni},
      year={2018},
      eprint={1805.01070},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
        }
        """,
        n_samples={"test": 8604},
        avg_character_length={"test": 70.06},
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_column("sentence", "text")
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["test"]
        )


class TopConstClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="TopConstClassification",
        description="This is a 20-class classification task, where the classes are given by the 19 most common top-constituent sequences in the corpus, plus a 20th category for all other structures.",
        reference="https://arxiv.org/abs/1805.01070",
        dataset={
            "path": "tasksource/linguisticprobing",
            "revision": "1372f0696870b8ad09ec3ed123173be722c7a891",
            "name": "top_constituents",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2018-01-01", "2018-07-08"),
        form=["written"],
        domains=["Non-fiction", "Fiction"],
        task_subtypes=["Topic classification"],
        license="BSD",
        socioeconomic_status="mixed",
        annotations_creators="expert-annotated",
        dialect=[],
        text_creation="found",
        bibtex_citation="""
        @misc{conneau2018cram,
      title={What you can cram into a single vector: Probing sentence embeddings for linguistic properties}, 
      author={Alexis Conneau and German Kruszewski and Guillaume Lample and Loïc Barrault and Marco Baroni},
      year={2018},
      eprint={1805.01070},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
        }
        """,
        n_samples={"test": 6866},
        avg_character_length={"test": 91.23},
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_column("sentence", "text")
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["test"]
        )


class BShiftClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="BShiftClassification",
        description="In this classification task the goal is to predict whether two consecutive tokens within the sentence have been inverted (label I for inversion and O for original).",
        reference="https://arxiv.org/abs/1805.01070",
        dataset={
            "path": "tasksource/linguisticprobing",
            "revision": "1372f0696870b8ad09ec3ed123173be722c7a891",
            "name": "bigram_shift",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2018-01-01", "2018-07-08"),
        form=["written"],
        domains=["Non-fiction", "Fiction"],
        task_subtypes=["Topic classification"],
        license="BSD",
        socioeconomic_status="mixed",
        annotations_creators="expert-annotated",
        dialect=[],
        text_creation="found",
        bibtex_citation="""
        @misc{conneau2018cram,
      title={What you can cram into a single vector: Probing sentence embeddings for linguistic properties}, 
      author={Alexis Conneau and German Kruszewski and Guillaume Lample and Loïc Barrault and Marco Baroni},
      year={2018},
      eprint={1805.01070},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
        }
        """,
        n_samples={"test": 1000},
        avg_character_length={"test": 62.76},
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_column("sentence", "text")
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["test"]
        )


class TenseClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="TenseClassification",
        description="This is a binary classification task, based on whether the main verb of the sentence is marked as being in the present (PRES class) or past (PAST class) tense",
        reference="https://arxiv.org/abs/1805.01070",
        dataset={
            "path": "tasksource/linguisticprobing",
            "revision": "1372f0696870b8ad09ec3ed123173be722c7a891",
            "name": "past_present",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2018-01-01", "2018-07-08"),
        form=["written"],
        domains=["Non-fiction", "Fiction"],
        task_subtypes=["Topic classification"],
        license="BSD",
        socioeconomic_status="mixed",
        annotations_creators="expert-annotated",
        dialect=[],
        text_creation="found",
        bibtex_citation="""
        @misc{conneau2018cram,
      title={What you can cram into a single vector: Probing sentence embeddings for linguistic properties}, 
      author={Alexis Conneau and German Kruszewski and Guillaume Lample and Loïc Barrault and Marco Baroni},
      year={2018},
      eprint={1805.01070},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
        }
        """,
        n_samples={"test": 8666},
        avg_character_length={"test": 69.21},
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_column("sentence", "text")
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["test"]
        )


class SubjNumClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SubjNumClassification",
        description="Another binary classification task, this time focusing on the number of the subject of the main clause. The classes are NN (singular) and NNS (plural or mass: 'personnel', 'clientele', etc). As the class labels suggest, only common nouns are considered.",
        reference="https://arxiv.org/abs/1805.01070",
        dataset={
            "path": "tasksource/linguisticprobing",
            "revision": "1372f0696870b8ad09ec3ed123173be722c7a891",
            "name": "subj_number",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2018-01-01", "2018-07-08"),
        form=["written"],
        domains=["Non-fiction", "Fiction"],
        task_subtypes=["Topic classification"],
        license="BSD",
        socioeconomic_status="mixed",
        annotations_creators="expert-annotated",
        dialect=[],
        text_creation="found",
        bibtex_citation="""
        @misc{conneau2018cram,
      title={What you can cram into a single vector: Probing sentence embeddings for linguistic properties}, 
      author={Alexis Conneau and German Kruszewski and Guillaume Lample and Loïc Barrault and Marco Baroni},
      year={2018},
      eprint={1805.01070},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
        }
        """,
        n_samples={"test": 8149},
        avg_character_length={"test": 85.73},
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_column("sentence", "text")
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["test"]
        )


class ObjNumClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="ObjNumClassification",
        description="This binary classification task is analogous to the one above, but this time focusing on the direct object of the main clause. The labels are again NN to represent the singular class and NNS for the plural/mass one.",
        reference="https://arxiv.org/abs/1805.01070",
        dataset={
            "path": "tasksource/linguisticprobing",
            "revision": "1372f0696870b8ad09ec3ed123173be722c7a891",
            "name": "obj_number",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2018-01-01", "2018-07-08"),
        form=["written"],
        domains=["Non-fiction", "Fiction"],
        task_subtypes=["Topic classification"],
        license="BSD",
        socioeconomic_status="mixed",
        annotations_creators="expert-annotated",
        dialect=[],
        text_creation="found",
        bibtex_citation="""
        @misc{conneau2018cram,
      title={What you can cram into a single vector: Probing sentence embeddings for linguistic properties}, 
      author={Alexis Conneau and German Kruszewski and Guillaume Lample and Loïc Barrault and Marco Baroni},
      year={2018},
      eprint={1805.01070},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
        }
        """,
        n_samples={"test": 7894},
        avg_character_length={"test": 84.74},
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_column("sentence", "text")
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["test"]
        )


class SOMOClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SOMOClassification",
        description="This binary task asks whether a sentence occurs as-is in the source corpus (O label, for Original), or whether a (single) randomly picked noun or verb was replaced with another form with the same part of speech (C label, for Changed).",
        reference="https://arxiv.org/abs/1805.01070",
        dataset={
            "path": "tasksource/linguisticprobing",
            "revision": "1372f0696870b8ad09ec3ed123173be722c7a891",
            "name": "odd_man_out",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2018-01-01", "2018-07-08"),
        form=["written"],
        domains=["Non-fiction", "Fiction"],
        task_subtypes=["Topic classification"],
        license="BSD",
        socioeconomic_status="mixed",
        annotations_creators="expert-annotated",
        dialect=[],
        text_creation="found",
        bibtex_citation="""
        @misc{conneau2018cram,
      title={What you can cram into a single vector: Probing sentence embeddings for linguistic properties}, 
      author={Alexis Conneau and German Kruszewski and Guillaume Lample and Loïc Barrault and Marco Baroni},
      year={2018},
      eprint={1805.01070},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
        }
        """,
        n_samples={"test": 8446},
        avg_character_length={"test": 92.36},
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_column("sentence", "text")
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["test"]
        )


class CoordInvClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="CoordInvClassification",
        description="Binary task asking to distinguish between original sentence (class O) and sentences where the order of two coordinated clausal conjoints has been inverted (class I).",
        reference="https://arxiv.org/abs/1805.01070",
        dataset={
            "path": "tasksource/linguisticprobing",
            "revision": "1372f0696870b8ad09ec3ed123173be722c7a891",
            "name": "coordination_inversion",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2018-01-01", "2018-07-08"),
        form=["written"],
        domains=["Non-fiction", "Fiction"],
        task_subtypes=["Topic classification"],
        license="BSD",
        socioeconomic_status="mixed",
        annotations_creators="expert-annotated",
        dialect=[],
        text_creation="found",
        bibtex_citation="""
        @misc{conneau2018cram,
      title={What you can cram into a single vector: Probing sentence embeddings for linguistic properties}, 
      author={Alexis Conneau and German Kruszewski and Guillaume Lample and Loïc Barrault and Marco Baroni},
      year={2018},
      eprint={1805.01070},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
        }
        """,
        n_samples={"test": 10002},
        avg_character_length={"test": 77.56},
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_column("sentence", "text")
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["test"]
        )
