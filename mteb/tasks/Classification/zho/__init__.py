from __future__ import annotations

from .CMTEBClassification import (
    IFlyTek,
    JDReview,
    MultilingualSentiment,
    OnlineShopping,
    TNews,
    Waimai,
)
from .FinChinaSentimentClassification import FinChinaSentimentClassification
from .FinFEClassification import FinFEClassification
from .FinNSPClassification import FinNSPClassification
from .OpenFinDataSentimentClassification import OpenFinDataSentimentClassification
from .Weibo21Classification import Weibo21Classification
from .YueOpenriceReviewClassification import YueOpenriceReviewClassification

__all__ = [
    "OpenFinDataSentimentClassification",
    "Weibo21Classification",
    "FinNSPClassification",
    "FinFEClassification",
    "FinChinaSentimentClassification",
    "IFlyTek",
    "JDReview",
    "MultilingualSentiment",
    "OnlineShopping",
    "TNews",
    "Waimai",
    "YueOpenriceReviewClassification",
]
