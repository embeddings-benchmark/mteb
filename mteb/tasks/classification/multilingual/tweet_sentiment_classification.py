from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata

_LANGUAGES = {
    "arabic": ["ara-Arab"],
    "english": ["eng-Latn"],
    "german": ["deu-Latn"],
    "french": ["fra-Latn"],
    "italian": ["ita-Latn"],
    "portuguese": ["por-Latn"],
    "spanish": ["spa-Latn"],
    "hindi": ["hin-Deva"],
}


class TweetSentimentClassification(AbsTaskClassification):
    fast_loading = True
    metadata = TaskMetadata(
        name="TweetSentimentClassification",
        dataset={
            "path": "mteb/tweet_sentiment_multilingual",
            "revision": "d522bb117c32f5e0207344f69f7075fc9941168b",
        },
        description="A multilingual Sentiment Analysis dataset consisting of tweets in 8 different languages.",
        reference="https://aclanthology.org/2022.lrec-1.27",
        category="t2c",
        modalities=["text"],
        type="Classification",
        eval_splits=["test"],
        eval_langs=_LANGUAGES,
        main_score="accuracy",
        date=("2018-05-01", "2020-03-31"),
        domains=["Social", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="cc-by-3.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{barbieri-etal-2022-xlm,
  address = {Marseille, France},
  author = {Barbieri, Francesco  and
Espinosa Anke, Luis  and
Camacho-Collados, Jose},
  booktitle = {Proceedings of the Thirteenth Language Resources and Evaluation Conference},
  month = jun,
  pages = {258--266},
  publisher = {European Language Resources Association},
  title = {{XLM}-{T}: Multilingual Language Models in {T}witter for Sentiment Analysis and Beyond},
  url = {https://aclanthology.org/2022.lrec-1.27},
  year = {2022},
}
""",
    )

    def dataset_transform(self):
        for lang in self.hf_subsets:
            self.dataset[lang] = self.stratified_subsampling(
                self.dataset[lang], n_samples=256, seed=self.seed, splits=["test"]
            )
