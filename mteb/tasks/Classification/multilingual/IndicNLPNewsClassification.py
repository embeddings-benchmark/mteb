from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.MultilingualTask import MultilingualTask
from mteb.abstasks.TaskMetadata import TaskMetadata

_LANGUAGES = {
    "gu": ["guj-Gujr"],
    "kn": ["kan-Knda"],
    "mal": ["mal-Mlym"],
    "mr": ["mar-Deva"],
    "tel": ["tel-Telu"],
    "ori": ["ori-Orya"],
    "pa": ["pan-Guru"],
    "ta": ["tam-Taml"],
}


class IndicNLPNewsClassification(MultilingualTask, AbsTaskClassification):
    metadata = TaskMetadata(
        name="IndicNLPNewsClassification",
        dataset={
            "path": "Sakshamrzt/IndicNLP-Multilingual",
            "revision": "3f23bd4a622a462adfb6989419cfadf7dc778f25",
        },
        description="A News classification dataset in multiple Indian regional languages.",
        reference="https://github.com/AI4Bharat/indicnlp_corpus#indicnlp-news-article-classification-dataset",
        type="Classification",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=_LANGUAGES,
        main_score="accuracy",
        date=("2020-09-01", "2022-04-09"),
        domains=["News", "Written"],
        dialect=[],
        task_subtypes=["Topic classification"],
        license="cc-by-nc-4.0",
        annotations_creators="expert-annotated",
        sample_creation="found",
        bibtex_citation="""
      @article{kunchukuttan2020indicnlpcorpus,
    title={AI4Bharat-IndicNLP Corpus: Monolingual Corpora and Word Embeddings for Indic Languages},
    author={Anoop Kunchukuttan and Divyanshu Kakwani and Satish Golla and Gokul N.C. and Avik Bhattacharyya and Mitesh M. Khapra and Pratyush Kumar},
    year={2020},
    journal={arXiv preprint arXiv:2005.00085}
}""",
    )

    def dataset_transform(self):
        for lang in self.hf_subsets:
            self.dataset[lang] = self.dataset[lang].rename_columns(
                {"news": "text", "class": "label"}
            )
            if lang == "pa":
                self.dataset[lang] = self.dataset[lang].remove_columns("headline")
            if self.dataset[lang]["test"].num_rows > 2048:
                self.dataset[lang] = self.stratified_subsampling(
                    self.dataset[lang],
                    seed=self.seed,
                    splits=["test"],
                )
