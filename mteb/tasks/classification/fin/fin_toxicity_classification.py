from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class FinToxicityClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="FinToxicityClassification",
        description="This dataset is a DeepL -based machine translated version of the Jigsaw toxicity dataset for Finnish. The dataset is originally from a Kaggle competition https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data. The original dataset poses a multi-label text classification problem and includes the labels identity_attack, insult, obscene, severe_toxicity, threat and toxicity. Here adapted for toxicity classification, which is the most represented class.",
        dataset={
            "path": "TurkuNLP/jigsaw_toxicity_pred_fi",
            "revision": "6e7340e6be87124f319e25290778760c14df64d3",
        },
        reference="https://aclanthology.org/2023.nodalida-1.68",
        type="Classification",
        category="t2c",
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
        bibtex_citation=r"""
@inproceedings{eskelinen-etal-2023-toxicity,
  author = {Eskelinen, Anni  and
Silvala, Laura  and
Ginter, Filip  and
Pyysalo, Sampo  and
Laippala, Veronika},
  booktitle = {Proceedings of the 24th Nordic Conference on Computational Linguistics (NoDaLiDa)},
  month = may,
  title = {Toxicity Detection in {F}innish Using Machine Translation},
  year = {2023},
}
""",
        superseded_by="FinToxicityClassification.v2",
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_column("label_toxicity", "label")
        remove_cols = [
            col
            for col in self.dataset["test"].column_names
            if col not in ["text", "label"]
        ]
        self.dataset = self.dataset.remove_columns(remove_cols)


class FinToxicityClassificationV2(AbsTaskClassification):
    metadata = TaskMetadata(
        name="FinToxicityClassification.v2",
        description="This dataset is a DeepL -based machine translated version of the Jigsaw toxicity dataset for Finnish. The dataset is originally from a Kaggle competition https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data. The original dataset poses a multi-label text classification problem and includes the labels identity_attack, insult, obscene, severe_toxicity, threat and toxicity. Here adapted for toxicity classification, which is the most represented class. This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)",
        dataset={
            "path": "mteb/fin_toxicity",
            "revision": "1deba6e874be1d5632a4ac0d1fb71f4bc3dea0d6",
        },
        reference="https://aclanthology.org/2023.nodalida-1.68",
        type="Classification",
        category="t2c",
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
        bibtex_citation=r"""
@inproceedings{eskelinen-etal-2023-toxicity,
  author = {Eskelinen, Anni  and
Silvala, Laura  and
Ginter, Filip  and
Pyysalo, Sampo  and
Laippala, Veronika},
  booktitle = {Proceedings of the 24th Nordic Conference on Computational Linguistics (NoDaLiDa)},
  month = may,
  title = {Toxicity Detection in {F}innish Using Machine Translation},
  year = {2023},
}
""",
        adapted_from=["FinToxicityClassification"],
    )
