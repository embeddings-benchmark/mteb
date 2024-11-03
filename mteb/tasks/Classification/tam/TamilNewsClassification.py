from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class TamilNewsClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="TamilNewsClassification",
        description="A Tamil dataset for 6-class classification of Tamil news articles",
        reference="https://github.com/vanangamudi/tamil-news-classification",
        dataset={
            "path": "mlexplorer008/tamil_news_classification",
            "revision": "bb34dd6690cf17aa731d75d45388c5801b8c4e4b",
        },
        type="Classification",
        category="s2s",
        modalities=["text"],
        date=("2014-01-01", "2018-01-01"),
        eval_splits=["test"],
        eval_langs=["tam-Taml"],
        main_score="f1",
        domains=["News", "Written"],
        task_subtypes=["Topic classification"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@article{kunchukuttan2020indicnlpcorpus,
    title={AI4Bharat-IndicNLP Corpus: Monolingual Corpora and Word Embeddings for Indic Languages},
    author={Anoop Kunchukuttan and Divyanshu Kakwani and Satish Golla and Gokul N.C. and Avik Bhattacharyya and Mitesh M. Khapra and Pratyush Kumar},
    year={2020},
    journal={arXiv preprint arXiv:2005.00085},
}""",
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_columns(
            {"NewsInTamil": "text", "Category": "label"}
        )
        self.dataset = self.stratified_subsampling(self.dataset, seed=self.seed)
