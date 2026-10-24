from .csfdsk_movie_review_sentiment_classification import (
    CSFDSKMovieReviewSentimentClassification,
    CSFDSKMovieReviewSentimentClassificationV2,
)
from .multi_eup_slovak_classification import (
    MultiEupSlovakGenderClassification,
    MultiEupSlovakPartyClassification,
)
from .slovak_hate_speech_classification import (
    SlovakHateSpeechClassification,
    SlovakHateSpeechClassificationV2,
)
from .slovak_movie_review_sentiment_classification import (
    SlovakMovieReviewSentimentClassification,
    SlovakMovieReviewSentimentClassificationV2,
)
from .slovak_parla_sent_classification import SlovakParlaSentClassification

__all__ = [
    "CSFDSKMovieReviewSentimentClassification",
    "CSFDSKMovieReviewSentimentClassificationV2",
    "MultiEupSlovakGenderClassification",
    "MultiEupSlovakPartyClassification",
    "SlovakHateSpeechClassification",
    "SlovakHateSpeechClassificationV2",
    "SlovakMovieReviewSentimentClassification",
    "SlovakMovieReviewSentimentClassificationV2",
    "SlovakParlaSentClassification",
]
