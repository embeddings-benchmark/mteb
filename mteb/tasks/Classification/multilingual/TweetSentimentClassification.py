from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import AbsTaskClassification, MultilingualTask

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


class TweetSentimentClassification(MultilingualTask, AbsTaskClassification):
    fast_loading = True
    metadata = TaskMetadata(
        name="TweetSentimentClassification",
        dataset={
            "path": "mteb/tweet_sentiment_multilingual",
            "revision": "d522bb117c32f5e0207344f69f7075fc9941168b",
        },
        description="A multilingual Sentiment Analysis dataset consisting of tweets in 8 different languages.",
        reference="https://aclanthology.org/2022.lrec-1.27",
        category="s2s",
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
        bibtex_citation="""
            @inproceedings{barbieri-etal-2022-xlm,
                title = "{XLM}-{T}: Multilingual Language Models in {T}witter for Sentiment Analysis and Beyond",
                author = "Barbieri, Francesco  and
                Espinosa Anke, Luis  and
                Camacho-Collados, Jose",
                booktitle = "Proceedings of the Thirteenth Language Resources and Evaluation Conference",
                month = jun,
                year = "2022",
                address = "Marseille, France",
                publisher = "European Language Resources Association",
                url = "https://aclanthology.org/2022.lrec-1.27",
                pages = "258--266",
                abstract = "Language models are ubiquitous in current NLP, and their multilingual capacity has recently attracted considerable attention. However, current analyses have almost exclusively focused on (multilingual variants of) standard benchmarks, and have relied on clean pre-training and task-specific corpora as multilingual signals. In this paper, we introduce XLM-T, a model to train and evaluate multilingual language models in Twitter. In this paper we provide: (1) a new strong multilingual baseline consisting of an XLM-R (Conneau et al. 2020) model pre-trained on millions of tweets in over thirty languages, alongside starter code to subsequently fine-tune on a target task; and (2) a set of unified sentiment analysis Twitter datasets in eight different languages and a XLM-T model trained on this dataset.",
            }
        """,
        descriptive_stats={
            "n_samples": {"test": 2048},
            "avg_character_length": {"test": 83.51},
        },
    )

    def dataset_transform(self):
        for lang in self.hf_subsets:
            self.dataset[lang] = self.stratified_subsampling(
                self.dataset[lang], n_samples=256, seed=self.seed, splits=["test"]
            )
