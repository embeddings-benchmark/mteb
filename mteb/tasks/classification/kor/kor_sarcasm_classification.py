from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class KorSarcasmClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="KorSarcasmClassification",
        description="The Korean Sarcasm Dataset was created to detect sarcasm in text, which can significantly alter the original meaning of a sentence. 9319 tweets were collected from Twitter and labeled for sarcasm or not_sarcasm. These tweets were gathered by querying for: irony sarcastic, and sarcasm. The dataset was created by gathering HTML data from Twitter. Queries for hashtags that include sarcasm and variants of it were used to return tweets. It was preprocessed by removing the keyword hashtag, urls and mentions of the user to preserve anonymity.",
        dataset={
            "path": "mteb/KorSarcasmClassification",
            "revision": "6701f384372c04aa8c64b10582e72eb84135a1d4",
        },
        reference="https://github.com/SpellOnYou/korean-sarcasm",
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs=["kor-Hang"],
        main_score="accuracy",
        date=("2018-10-31", "2019-09-28"),  # estimated based on git history
        domains=["Social", "Written"],
        task_subtypes=["Topic classification"],
        license="mit",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@misc{kim2019kocasm,
  author = {Kim, Jiwon and Cho, Won Ik},
  howpublished = {https://github.com/SpellOnYou/korean-sarcasm},
  journal = {GitHub repository},
  publisher = {GitHub},
  title = {Kocasm: Korean Automatic Sarcasm Detection},
  year = {2019},
}
""",
        superseded_by="KorSarcasmClassification.v2",
    )


class KorSarcasmClassificationV2(AbsTaskClassification):
    metadata = TaskMetadata(
        name="KorSarcasmClassification.v2",
        description="The Korean Sarcasm Dataset was created to detect sarcasm in text, which can significantly alter the original meaning of a sentence. 9319 tweets were collected from Twitter and labeled for sarcasm or not_sarcasm. These tweets were gathered by querying for: irony sarcastic, and sarcasm. The dataset was created by gathering HTML data from Twitter. Queries for hashtags that include sarcasm and variants of it were used to return tweets. It was preprocessed by removing the keyword hashtag, urls and mentions of the user to preserve anonymity. This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)",
        dataset={
            "path": "mteb/kor_sarcasm",
            "revision": "0e5e17b4dba569776e445f5639ba13dc406b2b0e",
        },
        reference="https://github.com/SpellOnYou/korean-sarcasm",
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["kor-Hang"],
        main_score="accuracy",
        date=("2018-10-31", "2019-09-28"),  # estimated based on git history
        domains=["Social", "Written"],
        task_subtypes=["Topic classification"],
        license="mit",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@misc{kim2019kocasm,
  author = {Kim, Jiwon and Cho, Won Ik},
  howpublished = {https://github.com/SpellOnYou/korean-sarcasm},
  journal = {GitHub repository},
  publisher = {GitHub},
  title = {Kocasm: Korean Automatic Sarcasm Detection},
  year = {2019},
}
""",
        adapted_from=["KorSarcasmClassification"],
    )

    def dataset_transform(self):
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["train"]
        )
