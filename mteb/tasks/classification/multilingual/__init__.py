from .afri_hate_classification import AfriHateClassification
from .afri_senti_classification import AfriSentiClassification
from .afri_senti_lang_classification import AfriSentiLangClassification
from .amazon_counterfactual_classification import AmazonCounterfactualClassification
from .amazon_reviews_classification import AmazonReviewsClassification
from .catalonia_tweet_classification import CataloniaTweetClassification
from .cyrillic_turkic_lang_classification import CyrillicTurkicLangClassification
from .hin_dialect_classification import HinDialectClassification
from .hume_multilingual_sentiment_classification import (
    HUMEMultilingualSentimentClassification,
)
from .indic_lang_classification import IndicLangClassification
from .indic_nlp_news_classification import IndicNLPNewsClassification
from .indic_sentiment_classification import IndicSentimentClassification
from .injongo_intent import InjongoIntent
from .kin_news_classification import KinNewsClassification
from .language_classification import LanguageClassification
from .m_in_ds14 import MInDS14Classification
from .masakha_news_classification import MasakhaNEWSClassification
from .massive_intent_classification import MassiveIntentClassification
from .massive_scenario_classification import MassiveScenarioClassification
from .mtop_domain_classification import MTOPDomainClassification
from .mtop_intent_classification import MTOPIntentClassification
from .multi_hate_classification import MultiHateClassification
from .multilingual_sentiment_classification import MultilingualSentimentClassification
from .naija_senti import NaijaSenti
from .nordic_lang_classification import NordicLangClassification
from .nusa_paragraph_emotion_classification import NusaParagraphEmotionClassification
from .nusa_paragraph_topic_classification import NusaParagraphTopicClassification
from .nusa_x_senti import NusaXSentiClassification
from .ru_nlu_intent_classification import RuNLUIntentClassification
from .ru_sci_bench_classification import (
    RuSciBenchCoreRiscClassification,
    RuSciBenchGRNTIClassificationV2,
    RuSciBenchOECDClassificationV2,
    RuSciBenchPubTypeClassification,
)
from .scala_classification import ScalaClassification
from .scandi_sent_classification import ScandiSentClassification
from .sib200_classification import SIB200Classification
from .sibfleurs import SIBFLEURSMultilingualClassification
from .south_african_lang_classification import SouthAfricanLangClassification
from .swiss_judgement_classification import SwissJudgementClassification
from .turkic_classification import TurkicClassification
from .tweet_sentiment_classification import TweetSentimentClassification
from .vox_populi_gender_id import VoxPopuliGenderID
from .vox_populi_language_id import VoxPopuliLanguageID

__all__ = [
    "AfriHateClassification",
    "AfriSentiClassification",
    "AfriSentiLangClassification",
    "AmazonCounterfactualClassification",
    "AmazonReviewsClassification",
    "CataloniaTweetClassification",
    "CyrillicTurkicLangClassification",
    "HUMEMultilingualSentimentClassification",
    "HinDialectClassification",
    "IndicLangClassification",
    "IndicNLPNewsClassification",
    "IndicSentimentClassification",
    "InjongoIntent",
    "KinNewsClassification",
    "LanguageClassification",
    "MInDS14Classification",
    "MTOPDomainClassification",
    "MTOPIntentClassification",
    "MasakhaNEWSClassification",
    "MassiveIntentClassification",
    "MassiveScenarioClassification",
    "MultiHateClassification",
    "MultilingualSentimentClassification",
    "NaijaSenti",
    "NordicLangClassification",
    "NusaParagraphEmotionClassification",
    "NusaParagraphTopicClassification",
    "NusaXSentiClassification",
    "RuNLUIntentClassification",
    "RuSciBenchCoreRiscClassification",
    "RuSciBenchGRNTIClassificationV2",
    "RuSciBenchOECDClassificationV2",
    "RuSciBenchPubTypeClassification",
    "SIB200Classification",
    "SIBFLEURSMultilingualClassification",
    "ScalaClassification",
    "ScandiSentClassification",
    "SouthAfricanLangClassification",
    "SwissJudgementClassification",
    "TurkicClassification",
    "TweetSentimentClassification",
    "VoxPopuliGenderID",
    "VoxPopuliLanguageID",
]
