"""Test if the task in MTEB doesn't contain common errors such as duplicates, train/test leakage etc.

These tests are not perfect, but should encourage contributors to re-examine the dataset.
"""

from typing import cast

from mteb.abstasks import AbsTask
from mteb.get_tasks import get_tasks
from mteb.types.statistics import DescriptiveStatistics, TextStatistics

# DO NOT ADD TO THIS LIST WITHOUT SPECIFYING WHY
KNOWN_ISSUES = {
    # initial addition: All existing issues
    "HotelReviewSentimentClassification",
    "OnlineStoreReviewSentimentClassification",
    "RestaurantReviewSentimentClassification",
    "TweetEmotionClassification",
    "TweetSarcasmClassification",
    "BengaliDocumentClassification",
    "BengaliHateSpeechClassification",
    "BengaliSentimentAnalysis",
    "CSFDCZMovieReviewSentimentClassification",
    "CzechProductReviewSentimentClassification",
    "CzechSoMeSentimentClassification",
    "AngryTweetsClassification",
    "DKHateClassification",
    "DanishPoliticalCommentsClassification",
    "Ddisco",
    "GermanPoliticiansTwitterSentimentClassification",
    "TenKGnadClassification",
    "ArxivClassification",
    "EmotionClassification",
    "FinancialPhrasebankClassification",
    "FrenkEnClassification",
    "HUMEEmotionClassification",
    "HUMEToxicConversationsClassification",
    "HUMETweetSentimentExtractionClassification",
    "ImdbClassification",
    "MAUDLegalBenchClassification",
    "OPP115DataSecurityLegalBenchClassification",
    "OPP115DoNotTrackLegalBenchClassification",
    "OPP115UserChoiceControlLegalBenchClassification",
    "OverrulingLegalBenchClassification",
    "PatentClassification",
    "SDSEyeProtectionClassification",
    "SDSGlovesClassification",
    "ToxicChatClassification",
    "ToxicConversationsClassification",
    "TweetSentimentExtractionClassification",
    "TweetTopicSingleClassification",
    "WikipediaBioMetChemClassification",
    "WikipediaChemFieldsClassification",
    "WikipediaCrystallographyAnalyticalClassification",
    "WikipediaTheoreticalAppliedClassification",
    "YahooAnswersTopicsClassification",
    "YelpReviewFullClassification",
    "EstonianValenceClassification",
    "DeepSentiPers",
    "NLPTwitterAnalysisClassification",
    "PerShopDomainClassification",
    "PerShopIntentClassification",
    "PersianTextEmotion",
    "SIDClassification",
    "SentimentDKSF",
    "SynPerTextToneClassification",
    "SynPerTextToneClassification.v3",
    "FilipinoHateSpeechClassification",
    "FinToxicityClassification",
    "FrenchBookReviews",
    "MovieReviewSentimentClassification",
    "MovieReviewSentimentClassification.v2",
    "GujaratiNewsClassification",
    "HebrewSentimentAnalysis",
    "HindiDiscourseClassification",
    "FrenkHrClassification",
    "IndonesianMongabayConservationClassification",
    "JavaneseIMDBClassification",
    "JapaneseSentimentClassification",
    "WRIMEClassification",
    "WRIMEClassification.v2",
    "KannadaNewsClassification",
    "KorFin",
    "KorSarcasmClassification",
    "KurdishSentimentClassification",
    "MalayalamNewsClassification",
    "MarathiNewsClassification",
    "MacedonianTweetSentimentClassification",
    "AfriSentiClassification",
    "AfriSentiLangClassification",
    "AmazonCounterfactualClassification",
    "AmazonReviewsClassification",
    "CataloniaTweetClassification",
    "HUMEMultilingualSentimentClassification",
    "HinDialectClassification",
    "IndicLangClassification",
    "IndicNLPNewsClassification",
    "IndicSentimentClassification",
    "LanguageClassification",
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
    "NusaX-senti",
    "RuNLUIntentClassification",
    "RuSciBenchCoreRiscClassification",
    "RuSciBenchGRNTIClassification.v2",
    "RuSciBenchOECDClassification.v2",
    "SIB200Classification",
    "ScandiSentClassification",
    "SouthAfricanLangClassification",
    "SwissJudgementClassification",
    "TurkicClassification",
    "TweetSentimentClassification",
    "MyanmarNews",
    "MyanmarNews.v2",
    "NepaliNewsClassification",
    "DutchGovernmentBiasClassification",
    "DutchNewsArticlesClassification",
    "IconclassClassification",
    "OpenTenderClassification",
    "VaccinChatNLClassification",
    "NoRecClassification",
    "NorwegianParliamentClassification",
    "NorwegianParliamentClassification.v2",
    "OdiaNewsClassification",
    "AllegroReviews",
    "CBD",
    "PAC",
    "PolEmo2.0-IN",
    "PolEmo2.0-OUT",
    "Moroco",
    "RomanianReviewsSentiment",
    "RomanianSentimentClassification",
    "GeoreviewClassification",
    "HeadlineClassification",
    "RuReviewsClassification",
    "SentiRuEval2016",
    "SinhalaNewsClassification",
    "SinhalaNewsSourceClassification",
    "CSFDSKMovieReviewSentimentClassification",
    "SlovakHateSpeechClassification",
    "SlovakMovieReviewSentimentClassification",
    "FrenkSlClassification",
    "SwahiliNewsClassification",
    "DalajClassification",
    "SweRecClassification",
    "SwedishSentimentClassification",
    "SwedishSentimentClassification.v2",
    "TamilNewsClassification",
    "TeluguAndhraJyotiNewsClassification",
    "WisesightSentimentClassification",
    "WisesightSentimentClassification.v2",
    "WongnaiReviewsClassification",
    "UkrFormalityClassification",
    "UrduRomanSentimentClassification",
    "AmazonPolarityVNClassification",
    "AmazonReviewsVNClassification",
    "Banking77VNClassification",
    "EmotionVNClassification",
    "ImdbVNClassification",
    "MTOPDomainVNClassification",
    "MTOPIntentVNClassification",
    "MassiveIntentVNClassification",
    "MassiveScenarioVNClassification",
    "ToxicConversationsVNClassification",
    "TweetSentimentExtractionVNClassification",
    "VieStudentFeedbackClassification",
    "IFlyTek",
    "IFlyTek.v2",
    "JDReview",
    "JDReview.v2",
    "MultilingualSentiment",
    "MultilingualSentiment.v2",
    "OnlineShopping",
    "TNews",
    "TNews.v2",
    "Waimai",
    "YueOpenriceReviewClassification",
    "BlurbsClusteringP2P",
    "BlurbsClusteringS2S",
    "BlurbsClusteringS2S.v2",
    "TenKGnadClusteringP2P",
    "TenKGnadClusteringS2S",
    "ArxivClusteringP2P",
    "ArxivClusteringP2P.v2",
    "ArxivClusteringS2S",
    "BiorxivClusteringP2P",
    "BiorxivClusteringP2P.v2",
    "BiorxivClusteringS2S",
    "BiorxivClusteringS2S.v2",
    "BuiltBenchClusteringP2P",
    "BuiltBenchClusteringS2S",
    "ClusTREC-Covid",
    "MedrxivClusteringP2P",
    "MedrxivClusteringP2P.v2",
    "MedrxivClusteringS2S",
    "MedrxivClusteringS2S.v2",
    "RedditClustering",
    "RedditClusteringP2P",
    "RedditClusteringP2P.v2",
    "RedditClustering.v2",
    "StackExchangeClustering",
    "StackExchangeClustering.v2",
    "StackExchangeClusteringP2P",
    "StackExchangeClusteringP2P.v2",
    "TwentyNewsgroupsClustering",
    "TwentyNewsgroupsClustering.v2",
    "WikiCitiesClustering",
    "BeytooteClustering",
    "NLPTwitterAnalysisClustering",
    "AlloProfClusteringS2S",
    "AlloProfClusteringS2S.v2",
    "HALClusteringS2S",
    "HALClusteringS2S.v2",
    "LivedoorNewsClustering",
    "MewsC16JaClustering",
    "IndicReviewsClusteringP2P",
    "MLSUMClusteringP2P",
    "MLSUMClusteringP2P.v2",
    "MLSUMClusteringS2S",
    "MLSUMClusteringS2S.v2",
    "MasakhaNEWSClusteringP2P",
    "MasakhaNEWSClusteringS2S",
    "SIB200ClusteringS2S",
    "WikiClusteringP2P.v2",
    "WikiClusteringP2P",
    "SNLClustering",
    "SNLHierarchicalClusteringS2S",
    "VGHierarchicalClusteringS2S",
    "PlscClusteringP2P",
    "PlscClusteringP2P.v2",
    "PlscClusteringS2S",
    "PlscClusteringS2S.v2",
    "RomaniBibleClustering",
    "SpanishNewsClusteringP2P",
    "SwednClustering",
    "SwednClusteringS2S",
    "SwednClusteringP2P",
    "RedditClusteringP2P-VN",
    "RedditClustering-VN",
    "StackExchangeClusteringP2P-VN",
    "StackExchangeClustering-VN",
    "TwentyNewsgroupsClustering-VN",
    "CLSClusteringS2S.v2",
    "CLSClusteringP2P",
    "CLSClusteringS2S",
    "ThuNewsClusteringP2P",
    "ThuNewsClusteringS2S",
    "AROCocoOrder",
    "AROFlickrOrder",
    "AROVisualAttribution",
    "AROVisualRelation",
    "ImageCoDe",
    "SugarCrepe",
    "Winoground",
    "EmitClassification",
    "KorHateSpeechMLClassification",
    "MultiEURLEXMultilabelClassification",
    "VABBMultiLabelClassification",
    "BrazilianToxicTweetsClassification",
    "CEDRClassification",
    "SwedishPatentCPCGroupClassification",
    "SwedishPatentCPCSubclassClassification",
    "RuSciBenchCitedCountRegression",
    "RuSciBenchYearPublRegression",
    # Add new datasets below with an explanation of why it is added
    # "name" # explanation
}


def _split_quality(
    name: str, split: str, split_stats: DescriptiveStatistics
) -> list[str]:
    errors = []

    num_samples = split_stats["num_samples"]  # type: ignore
    text_stats = split_stats.get("text_statistics", None)
    if text_stats:
        text_stats = cast(TextStatistics, text_stats)
        unique_texts = text_stats["unique_texts"]

        # Note: there could be cases where a dataset
        if num_samples != unique_texts:
            errors.append(
                f"{name} ({split}) contains text duplicates ({num_samples=}, {unique_texts=}), this can be intentional in multimodal datasets, but it likely unintentional."
            )

        min_text_length = text_stats["min_text_length"]
        if not (min_text_length > 3):
            errors.append(
                f"{name} ({split}) contains documents which are extremely short ({min_text_length=}), this likely indicate poor quality samples."
            )

        # Note: there could be cases where a dataset
        if num_samples != unique_texts:
            errors.append(
                f"{name} ({split}) contains duplicates ({num_samples=}, {unique_texts=})"
            )

    # train-test leakage
    number_texts_intersect_with_train = split_stats.get(
        "number_texts_intersect_with_train", None
    )
    if not (
        number_texts_intersect_with_train is None
        or number_texts_intersect_with_train == 0
    ):
        errors.append(
            f"{name} ({split}) has an overlap between train and test ({number_texts_intersect_with_train=})"
        )
    return errors


def _task_quality(task: AbsTask) -> list[str]:
    desc_stats = task.metadata.descriptive_stats

    errors = []
    if desc_stats is None:
        return []
    for split_name, split_stats in desc_stats.items():
        errors += _split_quality(task.metadata.name, split_name, split_stats)

    return errors


def test_dataset_quality():
    tasks = get_tasks(exclude_superseded=False, exclude_aggregate=True)

    errors: list[str] = []
    for task in tasks:
        if task.metadata.name in KNOWN_ISSUES:
            continue
        errors += _task_quality(task)

    if errors:
        raise AssertionError("\n".join([str(e) for e in errors]))
