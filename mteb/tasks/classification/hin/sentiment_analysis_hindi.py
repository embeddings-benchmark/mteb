from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class SentimentAnalysisHindi(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SentimentAnalysisHindi",
        description="Hindi Sentiment Analysis Dataset",
        reference="https://huggingface.co/datasets/OdiaGenAI/sentiment_analysis_hindi",
        dataset={
            "path": "OdiaGenAI/sentiment_analysis_hindi",
            "revision": "1beac1b941da76a9c51e3e5b39d230fde9a80983",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs=["hin-Deva"],
        main_score="f1",
        date=("2023-09-15", "2023-10-16"),
        dialect=[],
        domains=["Reviews", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="cc-by-nc-sa-4.0",
        annotations_creators="derived",
        sample_creation="found",
        bibtex_citation=r"""
@misc{OdiaGenAI,
  author = {Shantipriya Parida and Sambit Sekhar and Soumendra Kumar Sahoo and Swateek Jena and Abhijeet Parida and Satya Ranjan Dash and Guneet Singh Kohli},
  howpublished = {{https://huggingface.co/OdiaGenAI}},
  journal = {Hugging Face repository},
  publisher = {Hugging Face},
  title = {OdiaGenAI: Generative AI and LLM Initiative for the Odia Language},
  year = {2023},
}
""",
        superseded_by="SentimentAnalysisHindi.v2",
    )

    def dataset_transform(self):
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["train"]
        )


class SentimentAnalysisHindiV2(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SentimentAnalysisHindi.v2",
        description="Hindi Sentiment Analysis Dataset This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)",
        reference="https://huggingface.co/datasets/OdiaGenAI/sentiment_analysis_hindi",
        dataset={
            "path": "mteb/sentiment_analysis_hindi",
            "revision": "27fc099ce1c5a7295b9231e53a37648cdef6cb79",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["hin-Deva"],
        main_score="f1",
        date=("2023-09-15", "2023-10-16"),
        dialect=[],
        domains=["Reviews", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="cc-by-nc-sa-4.0",
        annotations_creators="derived",
        sample_creation="found",
        bibtex_citation=r"""
@misc{OdiaGenAI,
  author = {Shantipriya Parida and Sambit Sekhar and Soumendra Kumar Sahoo and Swateek Jena and Abhijeet Parida and Satya Ranjan Dash and Guneet Singh Kohli},
  howpublished = {{https://huggingface.co/OdiaGenAI}},
  journal = {Hugging Face repository},
  publisher = {Hugging Face},
  title = {OdiaGenAI: Generative AI and LLM Initiative for the Odia Language},
  year = {2023},
}
""",
        adapted_from=["SentimentAnalysisHindi"],
    )
