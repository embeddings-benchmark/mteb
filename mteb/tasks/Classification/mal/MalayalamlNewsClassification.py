from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class MalayalamNewsClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="MalayalamNewsClassification",
        dataset={
            "path": "Sakshamrzt/IndicNLP-Malayalam",
            "revision": "37ebc0fc6f40acf90b5053f70dc3b306f90de8e8",
        },
        description="A News classification dataset in Malayalam.",
        reference="https://github.com/AI4Bharat/indicnlp_corpus/blob/master/ai4bharat-indicnlp-corpus-2020.pdf",
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["mal-Mlym"],
        main_score="accuracy",
        date=("2020-09-01", "2022-04-09"),
        form=["written"],
        domains=["News"],
        dialect=[],
        task_subtypes=["Topic classification"],
        license="cc-by-nc-4.0",
        socioeconomic_status="medium",
        annotations_creators="expert-annotated",
        text_creation="found",
        bibtex_citation="""
      @article{kunchukuttan2020indicnlpcorpus,
    title={AI4Bharat-IndicNLP Corpus: Monolingual Corpora and Word Embeddings for Indic Languages},
    author={Anoop Kunchukuttan and Divyanshu Kakwani and Satish Golla and Gokul N.C. and Avik Bhattacharyya and Mitesh M. Khapra and Pratyush Kumar},
    year={2020},
    journal={arXiv preprint arXiv:2005.00085}
}""",
        n_samples={"test": 2048},
        avg_character_length={"test": 1929.6220703125},
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_columns({"news": "text", "class": "label"})
        self.dataset["train"] = self.dataset["train"].select(range(2048))
        self.dataset["test"] = self.dataset["test"].select(range(2048))
