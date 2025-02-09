from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class PunjabiNewsClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="PunjabiNewsClassification",
        description="A Punjabi dataset for 2-class classification of Punjabi news articles",
        reference="https://github.com/goru001/nlp-for-punjabi/",
        dataset={
            "path": "mlexplorer008/punjabi_news_classification",
            "revision": "cec3923e16519efe51d535497e711932b8f1dc44",
        },
        type="Classification",
        category="s2s",
        modalities=["text"],
        date=("2014-01-01", "2018-01-01"),
        eval_splits=["test"],
        eval_langs=["pan-Guru"],
        main_score="accuracy",
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
            {"article": "text", "is_about_politics": "label"}
        )
