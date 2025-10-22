from .dutch_book_review_sentiment_classification import (
    DutchBookReviewSentimentClassification,
    DutchBookReviewSentimentClassificationV2,
)
from .dutch_cola_classification import DutchColaClassification
from .dutch_government_bias_classification import DutchGovernmentBiasClassification
from .dutch_news_articles_classification import DutchNewsArticlesClassification
from .dutch_sarcastic_headlines_classification import (
    DutchSarcasticHeadlinesClassification,
)
from .iconclass_classification import IconclassClassification
from .open_tender_classification import OpenTenderClassification
from .vaccin_chat_nl_classification import VaccinChatNLClassification

__all__ = [
    "DutchBookReviewSentimentClassification",
    "DutchBookReviewSentimentClassificationV2",
    "DutchColaClassification",
    "DutchGovernmentBiasClassification",
    "DutchNewsArticlesClassification",
    "DutchSarcasticHeadlinesClassification",
    "IconclassClassification",
    "OpenTenderClassification",
    "VaccinChatNLClassification",
]
