from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class NepaliNewsClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="NepaliNewsClassification",
        description="A Nepali dataset for 7500 news articles ",
        reference="https://github.com/goru001/nlp-for-nepali",
        dataset={
            "path": "bpHigh/iNLTK_Nepali_News_Dataset",
            "revision": "79125f20d858a08f71ec4923169a6545221725c4",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        date=("2019-01-01", "2020-01-01"),
        eval_splits=["train"],
        eval_langs=["nep-Deva"],
        main_score="accuracy",
        domains=["News", "Written"],
        task_subtypes=["Topic classification"],
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{arora-2020-inltk,
  address = {Online},
  author = {Arora, Gaurav},
  booktitle = {Proceedings of Second Workshop for NLP Open Source Software (NLP-OSS)},
  doi = {10.18653/v1/2020.nlposs-1.10},
  editor = {Park, Eunjeong L.  and
Hagiwara, Masato  and
Milajevs, Dmitrijs  and
Liu, Nelson F.  and
Chauhan, Geeticka  and
Tan, Liling},
  month = nov,
  pages = {66--71},
  publisher = {Association for Computational Linguistics},
  title = {i{NLTK}: Natural Language Toolkit for Indic Languages},
  url = {https://aclanthology.org/2020.nlposs-1.10},
  year = {2020},
}
""",
        superseded_by="NepaliNewsClassification.v2",
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_column("paras", "text")
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["train"]
        )


class NepaliNewsClassificationV2(AbsTaskClassification):
    metadata = TaskMetadata(
        name="NepaliNewsClassification.v2",
        description="A Nepali dataset for 7500 news articles This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)",
        reference="https://github.com/goru001/nlp-for-nepali",
        dataset={
            "path": "mteb/nepali_news",
            "revision": "1e5e6cd30972f05f0f21af38bd3a887714d41938",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        date=("2019-01-01", "2020-01-01"),
        eval_splits=["test"],
        eval_langs=["nep-Deva"],
        main_score="accuracy",
        domains=["News", "Written"],
        task_subtypes=["Topic classification"],
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{arora-2020-inltk,
  address = {Online},
  author = {Arora, Gaurav},
  booktitle = {Proceedings of Second Workshop for NLP Open Source Software (NLP-OSS)},
  doi = {10.18653/v1/2020.nlposs-1.10},
  editor = {Park, Eunjeong L.  and
Hagiwara, Masato  and
Milajevs, Dmitrijs  and
Liu, Nelson F.  and
Chauhan, Geeticka  and
Tan, Liling},
  month = nov,
  pages = {66--71},
  publisher = {Association for Computational Linguistics},
  title = {i{NLTK}: Natural Language Toolkit for Indic Languages},
  url = {https://aclanthology.org/2020.nlposs-1.10},
  year = {2020},
}
""",
        adapted_from=["NepaliNewsClassification"],
    )

    def dataset_transform(self):
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["train"]
        )
