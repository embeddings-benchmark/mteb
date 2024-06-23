from __future__ import annotations

from mteb.abstasks import AbsTaskClassification
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
        date=("2014-01-01", "2018-01-01"),
        eval_splits=["test"],
        eval_langs=["ory-Orya"],
        main_score="f1",
        form=["written"],
        domains=["News"],
        task_subtypes=["Topic classification"],
        license="MIT",
        socioeconomic_status="mixed",
        annotations_creators="derived",
        dialect=[],
        text_creation="found",
        bibtex_citation="""@article{kunchukuttan2020indicnlpcorpus,
    title={AI4Bharat-IndicNLP Corpus: Monolingual Corpora and Word Embeddings for Indic Languages},
    author={Anoop Kunchukuttan and Divyanshu Kakwani and Satish Golla and Gokul N.C. and Avik Bhattacharyya and Mitesh M. Khapra and Pratyush Kumar},
    year={2020},
    journal={arXiv preprint arXiv:2005.00085},
}""",
        n_samples={"test": 2048},
        avg_character_length={"test": 49.24},
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_columns({"headings": "text"})
        self.dataset = self.stratified_subsampling(self.dataset, seed=self.seed)
