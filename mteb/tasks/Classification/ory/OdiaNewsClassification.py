from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class OdiaNewsClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="OdiaNewsClassification",
        description="A Odia dataset for 3-class classification of Odia news articles",
        reference="https://github.com/goru001/nlp-for-odia",
        dataset={
            "path": "mlexplorer008/odia_news_classification",
            "revision": "ffb8a34c9637fb20256e8c7be02504d16af4bd6b",
        },
        type="Classification",
        category="s2s",
        modalities=["text"],
        date=("2014-01-01", "2018-01-01"),
        eval_splits=["test"],
        eval_langs=["ory-Orya"],
        main_score="f1",
        domains=["News", "Written"],
        task_subtypes=["Topic classification"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@article{kunchukuttan2020indicnlpcorpus,
  author = {Anoop Kunchukuttan and Divyanshu Kakwani and Satish Golla and Gokul N.C. and Avik Bhattacharyya and Mitesh M. Khapra and Pratyush Kumar},
  journal = {arXiv preprint arXiv:2005.00085},
  title = {AI4Bharat-IndicNLP Corpus: Monolingual Corpora and Word Embeddings for Indic Languages},
  year = {2020},
}
""",
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_columns({"headings": "text"})
        self.dataset = self.stratified_subsampling(self.dataset, seed=self.seed)
