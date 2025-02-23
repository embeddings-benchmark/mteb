from __future__ import annotations

from mteb.abstasks.task_metadata import TaskMetadata
from mteb.abstasks.text.abs_text_classification import AbsTextClassification


class SentimentAnalysisHindi(AbsTextClassification):
    metadata = TaskMetadata(
        name="SentimentAnalysisHindi",
        description="Hindi Sentiment Analysis Dataset",
        reference="https://huggingface.co/datasets/OdiaGenAI/sentiment_analysis_hindi",
        dataset={
            "path": "OdiaGenAI/sentiment_analysis_hindi",
            "revision": "1beac1b941da76a9c51e3e5b39d230fde9a80983",
        },
        type="Classification",
        category="t2t",
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
        bibtex_citation="""@misc{OdiaGenAI, 
        author = {Shantipriya Parida and Sambit Sekhar and Soumendra Kumar Sahoo and Swateek Jena and Abhijeet Parida and Satya Ranjan Dash and Guneet Singh Kohli},  
        title = {OdiaGenAI: Generative AI and LLM Initiative for the Odia Language},  
        year = {2023},  
        publisher = {Hugging Face},  
        journal = {Hugging Face repository},  
        howpublished = {{https://huggingface.co/OdiaGenAI}}, } """,
    )

    def dataset_transform(self):
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["train"]
        )
