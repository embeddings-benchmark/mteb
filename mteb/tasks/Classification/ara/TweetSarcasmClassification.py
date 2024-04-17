from __future__ import annotations

from mteb.abstasks import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class TweetSarcasmClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="TweetSarcasmClassification",
        dataset={
            "path": "ar_sarcasm",
            "revision": "557bf94ac6177cc442f42d0b09b6e4b76e8f47c9",
        },
        description="Arabic sarcasm detection dataset, which was created through the reannotation of available Arabic sentiment analysis datasets.",
        reference="https://aclanthology.org/2020.osact-1.5/",
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["ara-Arab"],
        main_score="accuracy",
        date=("2020-01-01", "2021-01-01"),
        form=["written"],
        domains=["Social"],
        task_subtypes=["Sentiment/Hate speech"],
        license="MIT",
        socioeconomic_status="mixed",
        annotations_creators="human-annotated",
        dialect=["ara-arab-EG", "ara-arab-LB", "ara-arab-MA", "ara-arab-SA"],
        text_creation="found",
        bibtex_citation="""
@inproceedings{abu-farha-magdy-2020-arabic,
    title = "From {A}rabic Sentiment Analysis to Sarcasm Detection: The {A}r{S}arcasm Dataset",
    author = "Abu Farha, Ibrahim  and
      Magdy, Walid",
    editor = "Al-Khalifa, Hend  and
      Magdy, Walid  and
      Darwish, Kareem  and
      Elsayed, Tamer  and
      Mubarak, Hamdy",
    booktitle = "Proceedings of the 4th Workshop on Open-Source Arabic Corpora and Processing Tools, with a Shared Task on Offensive Language Detection",
    month = may,
    year = "2020",
    address = "Marseille, France",
    publisher = "European Language Resource Association",
    url = "https://aclanthology.org/2020.osact-1.5",
    pages = "32--39",
    abstract = "Sarcasm is one of the main challenges for sentiment analysis systems. Its complexity comes from the expression of opinion using implicit indirect phrasing. In this paper, we present ArSarcasm, an Arabic sarcasm detection dataset, which was created through the reannotation of available Arabic sentiment analysis datasets. The dataset contains 10,547 tweets, 16{\%} of which are sarcastic. In addition to sarcasm the data was annotated for sentiment and dialects. Our analysis shows the highly subjective nature of these tasks, which is demonstrated by the shift in sentiment labels based on annotators{'} biases. Experiments show the degradation of state-of-the-art sentiment analysers when faced with sarcastic content. Finally, we train a deep learning model for sarcasm detection using BiLSTM. The model achieves an F1 score of 0.46, which shows the challenging nature of the task, and should act as a basic baseline for future research on our dataset.",
    language = "English",
    ISBN = "979-10-95546-51-1",
}
""",
        n_samples={"test": 2110},
        avg_character_length={"test": 102.1},
    )

    def dataset_transform(self):
        # labels: 0 non-sarcastic, 1 sarcastic
        self.dataset = self.dataset.rename_columns({"tweet": "text", "sarcasm": "label"})
