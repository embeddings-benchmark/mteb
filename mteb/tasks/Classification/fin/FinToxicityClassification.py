from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class FinToxicityClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="FinToxicityClassification",
        description="""
        This dataset is a DeepL -based machine translated version of the Jigsaw toxicity dataset for Finnish. The dataset is originally from a Kaggle competition https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data.
        The original dataset poses a multi-label text classification problem and includes the labels identity_attack, insult, obscene, severe_toxicity, threat and toxicity.
        Here adapted for toxicity classification, which is the most represented class.
        """,
        dataset={
            "path": "TurkuNLP/jigsaw_toxicity_pred_fi",
            "revision": "6e7340e6be87124f319e25290778760c14df64d3",
        },
        reference="https://aclanthology.org/2023.nodalida-1.68",
        type="Classification",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fin-Latn"],
        main_score="f1",
        date=("2023-03-13", "2023-09-25"),
        domains=["News", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="machine-translated",
        bibtex_citation="""
        @inproceedings{eskelinen-etal-2023-toxicity,
            title = "Toxicity Detection in {F}innish Using Machine Translation",
            author = "Eskelinen, Anni  and
            Silvala, Laura  and
            Ginter, Filip  and
            Pyysalo, Sampo  and
            Laippala, Veronika",
            booktitle = "Proceedings of the 24th Nordic Conference on Computational Linguistics (NoDaLiDa)",
            month = may,
            year = "2023",
        }""",
        descriptive_stats={
            "n_samples": {"train": 2048, "test": 2048},
            "avg_character_length": {"train": 432.63, "test": 401.03},
        },
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_column("label_toxicity", "label")
        remove_cols = [
            col
            for col in self.dataset["test"].column_names
            if col not in ["text", "label"]
        ]
        self.dataset = self.dataset.remove_columns(remove_cols)
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["train", "test"]
        )
