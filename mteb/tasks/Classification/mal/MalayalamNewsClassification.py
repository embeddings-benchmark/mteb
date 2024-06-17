from __future__ import annotations

from mteb.abstasks import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class MalayalamNewsClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="MalayalamNewsClassification",
        description="A Malayalam dataset for 3-class classification of Malayalam news articles",
        reference="https://github.com/goru001/nlp-for-malyalam",
        dataset={
            "path": "mlexplorer008/malayalam_news_classification",
            "revision": "666f63bba2387456d8f846ea4d0565181bd47b81",
        },
        type="Classification",
        category="s2s",
        date=("2014-01-01", "2018-01-01"),
        eval_splits=["test"],
        eval_langs=["mal-Mlym"],
        main_score="accuracy",
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
        n_samples={"train": 5036, "test": 1260},
        avg_character_length={"train": 79.48, "test": 80.44},
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_columns({"headings": "text"})
