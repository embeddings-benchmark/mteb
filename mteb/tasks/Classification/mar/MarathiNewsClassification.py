from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class MarathiNewsClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="MarathiNewsClassification",
        description="A Marathi dataset for 3-class classification of Marathi news articles",
        reference="https://github.com/goru001/nlp-for-marathi",
        dataset={
            "path": "mlexplorer008/marathi_news_classification",
            "revision": "7640cf8132cca1f99995ac71512a670e3c965cf1",
        },
        type="Classification",
        category="s2s",
        modalities=["text"],
        date=("2014-01-01", "2018-01-01"),
        eval_splits=["test"],
        eval_langs=["mar-Deva"],
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
        self.dataset = self.dataset.rename_columns({"headline": "text"})
        self.dataset = self.stratified_subsampling(self.dataset, seed=self.seed)
