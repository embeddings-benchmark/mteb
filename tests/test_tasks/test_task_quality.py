"""Test if the task in MTEB doesn't contain common errors such as duplicates, train/test leakage etc.

These tests are not perfect, but should encourage contributors to re-examine the dataset.
"""

import math
import warnings
from typing import Any, cast

from mteb.abstasks import AbsTask
from mteb.get_tasks import get_tasks
from mteb.types.statistics import (
    AudioStatistics,
    ImageStatistics,
    LabelStatistics,
    RelevantDocsStatistics,
    ScoreStatistics,
    SplitDescriptiveStatistics,
    TextStatistics,
    TopRankedStatistics,
    VideoStatistics,
)

KNOWN_ISSUES: dict[str, list[str]] = {
    "short_text": [
        "ARCChallenge",
        "AVMemeExamVideoAudioCentricQA",
        "AVMemeExamVideoCentricQA",
        "AVQAVideoAudioCentricQA",
        "AVQAVideoCentricQA",
        "AfriSentiClassification",
        "AlphaNLI",
        "AmazonPolarityVNClassification",
        "AmazonReviewsVNClassification",
        "AngryTweetsClassification",
        "ArguAna",
        "ArguAna-Fa",
        "ArguAna-NL",
        "ArguAna-NL.v2",
        "ArguAna-PL",
        "ArxivClusteringS2S",
        "BLINKIT2TMultiChoice",
        "BLINKIT2TRetrieval",
        "BQ",
        "BSARDRetrieval",
        "BSARDRetrieval.v2",
        "BengaliDocumentClassification",
        "BibleNLPBitextMining",
        "BlurbsClusteringS2S",
        "BlurbsClusteringS2S.v2",
        "BornholmBitextMining",
        "BrightBiologyRetrieval",
        "BrightEarthScienceRetrieval",
        "BrightEconomicsRetrieval",
        "BrightPsychologyRetrieval",
        "BrightRetrieval",
        "BrightRoboticsRetrieval",
        "BrightStackoverflowRetrieval",
        "BrightSustainableLivingRetrieval",
        "BuiltBenchClusteringS2S",
        "CExaPPC",
        "CLSClusteringP2P",
        "CLSClusteringS2S",
        "CLSClusteringS2S.v2",
        "CMedQAv1-reranking",
        "COIRCodeSearchNetRetrieval",
        "CQADupstackEnglishRetrieval-Fa",
        "CQADupstackGamingRetrieval-Fa",
        "CQADupstackMathematicaRetrieval-Fa",
        "CQADupstackPhysicsRetrieval-Fa",
        "CQADupstackStatsRetrieval-Fa",
        "CQADupstackTexRetrieval-Fa",
        "CQADupstackUnixRetrieval-Fa",
        "CQADupstackWebmastersRetrieval-Fa",
        "CQADupstackWordpressRetrieval-Fa",
        "CSFDCZMovieReviewSentimentClassification",
        "CSFDSKMovieReviewSentimentClassification",
        "CVBenchCount",
        "CVBenchDepth",
        "CVBenchDistance",
        "ClimateFEVER",
        "ClimateFEVER-Fa",
        "ClimateFEVER-NL",
        "ClimateFEVER-VN",
        "ClimateFEVERHardNegatives",
        "ClimateFEVERHardNegatives.v2",
        "ClothoA2TRetrieval",
        "ClothoT2ARetrieval",
        "CmedqaRetrieval",
        "Cmnli",
        "CodeFeedbackMT",
        "CodeFeedbackST",
        "CodeRAGLibraryDocumentationSolutions",
        "CodeRAGOnlineTutorials",
        "CodeRAGProgrammingSolutions",
        "CodeRAGStackoverflowPosts",
        "CodeSearchNetRetrieval",
        "CommonVoiceMini17A2TRetrieval",
        "CommonVoiceMini17T2ARetrieval",
        "CommonVoiceMini21A2TRetrieval",
        "CommonVoiceMini21T2ARetrieval",
        "CovidQA",
        "CovidRetrieval",
        "CzechProductReviewSentimentClassification",
        "CzechSoMeSentimentClassification",
        "DBPedia-Fa",
        "DBPedia-PL",
        "DBPedia-PLHardNegatives",
        "DKHateClassification",
        "DanishMedicinesAgencyBitextMining",
        "DanishPoliticalCommentsClassification",
        "DiaBlaBitextMining",
        "DuRetrieval",
        "DutchNewsArticlesRetrieval",
        "EDIST2ITRetrieval",
        "ESCIReranking",
        "EcomRetrieval",
        "EmitClassification",
        "EmotionAnalysisPlus",
        "FEVER",
        "FEVER-NL",
        "FEVER-VN",
        "FEVERHardNegatives",
        "FEVERHardNegatives.v2",
        "FaithDial",
        "FalseFriendsGermanEnglish",
        "FiQA-PL",
        "FiQA2018",
        "FiQA2018-Fa",
        "FiQA2018-NL",
        "FilipinoHateSpeechClassification",
        "FinParaSTS",
        "FinToxicityClassification",
        "FrenchBookReviews",
        "FrenkEnClassification",
        "FrenkHrClassification",
        "FrenkSlClassification",
        "GeoreviewClassification",
        "GerDaLIR",
        "GermanPoliticiansTwitterSentimentClassification",
        "GreenNodeTableMarkdownRetrieval",
        "HALClusteringS2S",
        "HUMECore17InstructionReranking",
        "HUMEMultilingualSentimentClassification",
        "HUMENews21InstructionReranking",
        "HUMEToxicConversationsClassification",
        "HUMETweetSentimentExtractionClassification",
        "HatefulMemesI2TRetrieval",
        "HatefulMemesT2IRetrieval",
        "HeadlineClassification",
        "HebrewSentimentAnalysis",
        "HellaSwag",
        "HiFiTTSA2TRetrieval",
        "HiFiTTST2ARetrieval",
        "HindiDiscourseClassification",
        "HotpotQA-Fa",
        "HumanConceptsClustering",  # single-word concept items (e.g. "Bat", "Cat") are intentionally short by design
        "IFIRFiQA",
        "IFIRFire",
        "IN22ConvBitextMining",
        "IWSLT2017BitextMining",
        "IconclassClassification",
        "ImageCoDe",
        "ImageCoDeT2IRetrieval",
        "IndicCrosslingualSTS",
        "IndicLangClassification",
        "IndicNLPNewsClassification",
        "JDReview",
        "JDReview.v2",
        "JaGovFaqsRetrieval",
        "JamAltLyricA2TRetrieval",
        "JamAltLyricT2ARetrieval",
        "JinaVDRGitHubReadmeRetrieval",
        "KorHateSpeechMLClassification",
        "KorNLI",
        "KorSarcasmClassification",
        "LASSA2TRetrieval",
        "LASST2ARetrieval",
        "LanguageClassification",
        "LibriTTSA2TRetrieval",
        "LibriTTST2ARetrieval",
        "LinceMTBitextMining",
        "LitSearchRetrieval",
        "LivedoorNewsClustering",
        "LoTTE",
        "MELDAudioVideoZeroShot",
        "MELDVideoZeroShot",
        "MIRACLRetrieval",
        "MIRACLRetrievalHardNegatives",
        "MIRACLRetrievalHardNegatives.v2",
        "MKQARetrieval",
        "MLQuestions",
        "MMDocIRT2ITRetrieval",
        "MMVUVideoCentricQA",
        "MMarcoRetrieval",
        "MSMARCO",
        "MSMARCO-Fa",
        "MSMARCO-PL",
        "MSMARCO-VN",
        "MTOPDomainClassification",
        "MTOPDomainVNClassification",
        "MTOPIntentClassification",
        "MTOPIntentVNClassification",
        "MasakhaNEWSClassification",
        "MassiveIntentClassification",
        "MassiveIntentVNClassification",
        "MassiveScenarioClassification",
        "MassiveScenarioVNClassification",
        "MedicalRetrieval",
        "MemotionI2TRetrieval",
        "MintakaRetrieval",
        "MovieReviewSentimentClassification",
        "MrTidyRetrieval",
        "MultiHateClassification",
        "MultiLongDocReranking",
        "MultiLongDocRetrieval",
        "MultilingualNanoDBPediaRetrieval",
        "MultilingualNanoFiQA2018Retrieval",
        "MultilingualNanoMSMARCORetrieval",
        "MultilingualNanoNFCorpusRetrieval",
        "MultilingualNanoNQRetrieval",
        "MultilingualNanoQuoraRetrieval",
        "MultilingualNanoSCIDOCSRetrieval",
        "MultilingualNanoTouche2020Retrieval",
        "MultilingualSentimentClassification",
        "MyanmarNews",
        "MyanmarNews.v2",
        "NExTQAVideoCentricQA",
        "NFCorpus",
        "NFCorpus-Fa",
        "NFCorpus-NL",
        "NFCorpus-NL.v2",
        "NFCorpus-PL",
        "NFCorpus-VN",
        "NLPTwitterAnalysisClassification",
        "NLPTwitterAnalysisClustering",
        "NQ-Fa",
        "NTREXBitextMining",
        "NamaaMrTydiReranking",
        "NanoDBPediaRetrieval",
        "NanoFiQA2018Retrieval",
        "NanoMSMARCO-VN",
        "NanoNQRetrieval",
        "NanoQuoraRetrieval",
        "NanoSCIDOCSRetrieval",
        "NanoTouche2020Retrieval",
        "NeuCLIR2022Retrieval",
        "NeuCLIR2022RetrievalHardNegatives",
        "NeuCLIR2023Retrieval",
        "NeuCLIR2023RetrievalHardNegatives",
        "News21InstructionRetrieval",
        "NoRecClassification",
        "NollySentiBitextMining",
        "NorQuadRetrieval",
        "NovelQA",
        "OVENIT2TRetrieval",
        "Ocnli",
        "OdiaNewsClassification",
        "OmniVideoBenchVideoAudioCentricQA",
        "OmniVideoBenchVideoCentricQA",
        "OnlineShopping",
        "OnlineStoreReviewSentimentClassification",
        "OpenTenderRetrieval",
        "OverrulingLegalBenchClassification",
        "PAWSX",
        "PIQA",
        "ParsinluEntail",
        "PawsXPairClassification",
        "PeerQA",
        "PerceptionTestVideoAudioCentricQA",
        "PerceptionTestVideoCentricQA",
        "PersianTextEmotion",
        "PersianWebDocumentRetrieval",
        "PhincBitextMining",
        "PolEmo2.0-IN",
        "PolEmo2.0-OUT",
        "PubChemSMILESBitextMining",
        "PubChemSynonymPC",
        "QBQTC",
        "Quail",
        "Query2Query",
        "Quora-NL",
        "Quora-PL",
        "Quora-PLHardNegatives",
        "QuoraRetrieval",
        "QuoraRetrieval-Fa",
        "QuoraRetrieval-Fa.v2",
        "QuoraRetrievalHardNegatives",
        "QuoraRetrievalHardNegatives.v2",
        "R2MEDBiologyRetrieval",
        "RestaurantReviewSentimentClassification",
        "RiaNewsRetrieval",
        "RiaNewsRetrievalHardNegatives",
        "RiaNewsRetrievalHardNegatives.v2",
        "RomanianReviewsSentiment",
        "RomanianSentimentClassification",
        "RuBQReranking",
        "RuBQRetrieval",
        "RuNLUIntentClassification",
        "RuReviewsClassification",
        "SCIDOCS-Fa",
        "SDSKoPubVDRT2ITRetrieval",
        "SIQA",
        "SNLRetrieval",
        "STS12",
        "STS22",
        "STSB",
        "STSBenchmarkMultilingualSTS",
        "SWEbenchCodeRetrieval",
        "SentimentDKSF",
        "SinhalaNewsClassification",
        "SlovakMovieReviewSentimentClassification",
        "SpanishNewsClusteringP2P",
        "SpeechCommandsZeroshotv0.01",
        "SpeechCommandsZeroshotv0.02",
        "SummEvalFr",  # a genuinely empty machine-generated summary is present in the upstream data
        "SummEvalFrSummarization.v2",  # same, this is the .v2 revision of SummEvalFr
        "SweRecClassification",
        "SwedishPatentCPCGroupClassification",
        "SwedishPatentCPCSubclassClassification",
        "SwedishSentimentClassification",
        "SwednClusteringS2S",
        "SynPerChatbotRAGTopicsRetrieval",
        "SynPerChatbotTopicsRetrieval",
        "SynPerQARetrieval",
        "T2Reranking",
        "T2Retrieval",
        "TNews",
        "TNews.v2",
        "TRECCOVID",
        "TRECCOVID-Fa",
        "TRECCOVID-NL",
        "TRECCOVID-PL",
        "TRECDL2019",
        "TRECDL2020",
        "TUBerlinT2IRetrieval",
        "TalemaaderPC",
        "Tatoeba",
        "TempReasonL2Context",
        "TempReasonL2Fact",
        "TempReasonL2Pure",
        "TempReasonL3Context",
        "TempReasonL3Fact",
        "TempReasonL3Pure",
        "ThuNewsClusteringS2S",
        "Touche2020",
        "Touche2020-Fa",
        "Touche2020-Fa.v2",
        "Touche2020-NL",
        "Touche2020-PL",
        "Touche2020-VN",
        "ToxicConversationsClassification",
        "ToxicConversationsVNClassification",
        "TurHistQuadRetrieval",
        "TweetEmotionClassification",
        "TweetSentimentClassification",
        "TweetSentimentExtractionClassification",
        "TweetSentimentExtractionVNClassification",
        "UrduRomanSentimentClassification",
        "VALOR32KAT2VRetrieval",
        "VALOR32KT2VARetrieval",
        "VALOR32KT2VRetrieval",
        "VALOR32KV2TRetrieval",
        "VALOR32KVA2TRetrieval",
        "VALOR32KVT2ARetrieval",
        "VQA2IT2TRetrieval",
        "VaccinChatNLClassification",
        "VideoConPairClassification",
        "VideoRetrieval",
        "Vidore3ComputerScienceRetrieval.v2",
        "Vidore3EnergyRetrieval.v2",
        "Vidore3FinanceEnRetrieval.v2",
        "Vidore3FinanceFrRetrieval.v2",
        "Vidore3HrRetrieval.v2",
        "Vidore3IndustrialRetrieval.v2",
        "VisualNewsI2TRetrieval",
        "VisualNewsT2IRetrieval",
        "VizWizIT2TRetrieval",
        "VoyageMMarcoReranking",
        "WITT2IRetrieval",
        "WRIMEClassification",
        "WRIMEClassification.v2",
        "WebQAT2ITRetrieval",
        "WikiClusteringP2P",
        "WikiClusteringP2P.v2",
        "WinoGrande",
        "WisesightSentimentClassification",
        "WisesightSentimentClassification.v2",
        "XGlueWPRReranking",
        "XM3600T2IRetrieval",
        "XMarket",
        "XNLI",
        "XPQARetrieval",
        "YahooAnswersTopicsClassification",
        "YelpReviewFullClassification",
        "YueOpenriceReviewClassification",
        "mMARCO-NL",
    ],
    "duplicate_text": [
        "AfriHateClassification",
        "AfriSentiClassification",
        "AllegroReviews",
        "AlloprofReranking",
        "AmazonCounterfactualClassification",
        "AmazonReviewsClassification",
        "ArguAna",
        "ArguAna-NL",
        "ArguAna-NL.v2",
        "ArguAna-PL",
        "ArxivClassification",
        "ArxivClusteringP2P",
        "ArxivClusteringP2P.v2",
        "ArxivClusteringS2S",
        "AskUbuntuDupQuestions",
        "AskUbuntuDupQuestions-VN",
        "AudioCapsA2TRetrieval",
        "AudioCapsT2ARetrieval",
        "AudioSetStrongA2TRetrieval",
        "AudioSetStrongT2ARetrieval",
        "BLINKIT2IRetrieval",
        "BLINKIT2TRetrieval",
        "BSARDRetrieval.v2",
        "Banking77VNClassification",
        "BengaliHateSpeechClassification",
        "BengaliSentimentAnalysis",
        "BeytooteClustering",
        "BiorxivClusteringP2P",
        "BiorxivClusteringS2S",
        "BlurbsClusteringP2P",
        "BlurbsClusteringS2S",
        "BlurbsClusteringS2S.v2",
        "BuiltBenchClusteringP2P",
        "BuiltBenchClusteringS2S",
        "BuiltBenchReranking",
        "CASTELLAAMRRetrieval",
        "CMUArcticA2TRetrieval",
        "CMUArcticT2ARetrieval",
        "CMedQAv1-reranking",
        "CMedQAv2-reranking",
        "ClimateFEVER.v2",
        "ClusTREC-Covid",
        "CodeFeedbackST",
        "CodeSearchNetCCRetrieval",
        "CodeSearchNetRetrieval",
        "CodeTransOceanDL",
        "Core17InstructionRetrieval",
        "CosQA",
        "CzechProductReviewSentimentClassification",
        "CzechSoMeSentimentClassification",
        "DKHateClassification",
        "DS1000Retrieval",
        "DalajClassification",
        "DanishPoliticalCommentsClassification",
        "DeepSentiPers",
        "EDIST2ITRetrieval",
        "ESCIReranking",
        "EmitClassification",
        "EmoVDBA2TRetrieval",
        "EmoVDBT2ARetrieval",
        "EmotionAnalysisPlus",
        "EncyclopediaVQAIT2ITRetrieval",
        "EuroPIRQRetrieval",
        "FEVER",
        "FEVER-NL",
        "FEVER-VN",
        "FaithDial",
        "FiQA2018-VN",
        "FilipinoHateSpeechClassification",
        "FrenchBookReviews",
        "FrenkEnClassification",
        "FrenkHrClassification",
        "GigaSpeechA2TRetrieval",
        "GigaSpeechT2ARetrieval",
        "HebrewSentimentAnalysis",
        "HiFiTTSA2TRetrieval",
        "HiFiTTST2ARetrieval",
        "HinDialectClassification",
        "HumanConceptsClustering",  # small, fixed vocabulary of concept words repeats across categories
        "IFIRNFCorpus",
        "IFIRScifact",
        "IFlyTek",
        "IndicNLPNewsClassification",
        "IndonesianMongabayConservationClassification",
        "InfoSeekIT2ITRetrieval",
        "InfoSeekIT2TRetrieval",
        "JDReview",
        "JQaRAReranking",
        "JaCWIRReranking",
        "JaCWIRRetrieval",
        "JaCWIRRetrievalLite",
        "JaGovFaqsRetrieval",
        "KinNewsClassification",
        "KorFin",
        "KurdishSentimentClassification",
        "LASSA2TRetrieval",
        "LASST2ARetrieval",
        "LLaVAIT2TRetrieval",
        "LanguageClassification",
        "LegalQANLRetrieval",
        "LoTTE",
        "LocBenchRR",
        "MACSA2TRetrieval",
        "MACST2ARetrieval",
        "MAUDLegalBenchClassification",
        "MIRACLReranking",
        "MIRACLRetrieval",
        "MIRACLRetrievalHardNegatives",
        "MIRACLRetrievalHardNegatives.v2",
        "MKQARetrieval",
        "MLQARetrieval",
        "MLSUMClusteringP2P",
        "MLSUMClusteringS2S",
        "MLSUMClusteringS2S.v2",
        "MMVUVideoCentricQA",
        "MMarcoReranking",
        "MSMARCOv2",
        "MTOPDomainClassification",
        "MTOPDomainVNClassification",
        "MTOPIntentClassification",
        "MTOPIntentVNClassification",
        "MalayalamNewsClassification",
        "MasakhaNEWSClassification",
        "MassiveIntentClassification",
        "MassiveIntentVNClassification",
        "MassiveScenarioClassification",
        "MassiveScenarioVNClassification",
        "MedicalQARetrieval",
        "MedrxivClusteringP2P",
        "MedrxivClusteringP2P.v2",
        "MedrxivClusteringS2S",
        "MedrxivClusteringS2S.v2",
        "MindSmallReranking",  # num_documents is a flattened per-query candidate pool; the same article legitimately appears across many queries
        "MintakaRetrieval",
        "Moroco",
        "MrTidyRetrieval",
        "MultiSWEbenchRR",
        "MultilingualNanoDBPediaRetrieval",
        "MultilingualNanoNFCorpusRetrieval",
        "MultilingualNanoNQRetrieval",
        "MultilingualNanoQuoraRetrieval",
        "MultilingualNanoSCIDOCSRetrieval",
        "MultilingualNanoTouche2020Retrieval",
        "MultilingualSentiment",
        "MultilingualSentimentClassification",
        "NFCorpus",
        "NFCorpus-Fa",
        "NFCorpus-NL",
        "NFCorpus-NL.v2",
        "NFCorpus-PL",
        "NaijaSenti",
        "NanoFEVER-VN",
        "NanoNFCorpusRetrieval",
        "NanoNQRetrieval",
        "NanoSCIDOCSRetrieval",
        "NanoTouche2020Retrieval",
        "NevIR",
        "NordicLangClassification",
        "OKVQAIT2TRetrieval",
        "OVENIT2ITRetrieval",
        "OVENIT2TRetrieval",
        "OnlineStoreReviewSentimentClassification",
        "OpenTenderRetrieval",
        "PIQA",
        "PerShopDomainClassification",
        "PerShopIntentClassification",
        "PlscClusteringP2P",
        "PlscClusteringP2P.v2",
        "PlscClusteringS2S",
        "PlscClusteringS2S.v2",
        "Quora-NL",
        "Quora-PL",
        "Quora-PLHardNegatives",
        "QuoraRetrieval-Fa.v2",
        "R2MEDBiologyRetrieval",
        "RedditClustering",
        "RedditClustering-VN",
        "RedditClustering.v2",
        "RedditClusteringP2P",
        "RedditClusteringP2P-VN",
        "RedditClusteringP2P.v2",
        "RomanianSentimentClassification",
        "RuBQReranking",
        "RuNLUIntentClassification",
        "SDSEyeProtectionClassification",
        "SDSGlovesClassification",
        "SIQA",
        "SWEPolyBenchRR",
        "SWEbenchCodeRetrieval",
        "SWEbenchLiteRR",
        "SWEbenchMultilingualRR",
        "SWEbenchVerifiedRR",
        "ScandiSentClassification",
        "SciDocsRR-VN",
        "SentiRuEval2016",
        "SentimentDKSF",
        "SinhalaNewsClassification",
        "SlovakHateSpeechClassification",
        "SouthAfricanLangClassification",
        "SpanishNewsClusteringP2P",
        "SpanishPassageRetrievalS2P",
        "SpanishPassageRetrievalS2S",
        "SpartQA",
        "SpokenSQuADT2ARetrieval",
        "StackExchangeClustering",
        "StackExchangeClustering-VN",
        "StackExchangeClusteringP2P",
        "StackExchangeClusteringP2P-VN",
        "StackExchangeClusteringP2P.v2",
        "StackOverflowDupQuestions",
        "StackOverflowDupQuestions-VN",
        "StatcanDialogueDatasetRetrieval",
        "SwahiliNewsClassification",
        "SwedishSentimentClassification",
        "SwednClusteringS2S",
        "SwednRetrieval",
        "SwissJudgementClassification",
        "SyntecReranking",
        "SyntheticText2SQL",
        "T2Reranking",
        "TNews",
        "TRECCOVID-Fa.v2",
        "TRECCOVID-PL",
        "TamilNewsClassification",
        "TenKGnadClusteringP2P",
        "TenKGnadClusteringS2S",
        "Touche2020-VN",
        "ToxicChatClassification",
        "TurkicClassification",
        "TweetSarcasmClassification",
        "TweetSentimentExtractionVNClassification",
        "TwentyNewsgroupsClustering",
        "TwentyNewsgroupsClustering-VN",
        "TwentyNewsgroupsClustering.v2",
        "UrbanSound8KA2TRetrieval",
        "UrbanSound8KT2ARetrieval",
        "VABBRetrieval",
        "VQA2IT2TRetrieval",
        "VizWizIT2TRetrieval",
        "VoyageMMarcoReranking",
        "WebFAQRetrieval",
        "WebLINXCandidatesReranking",
        "WebQAT2ITRetrieval",
        "WikiClusteringP2P",
        "WikiClusteringP2P.v2",
        "WikiSQLRetrieval",
        "XM3600T2IRetrieval",
        "XMarket",
        "XPQARetrieval",
        "YahooAnswersTopicsClassification",
        "YueOpenriceReviewClassification",
        "ZacLegalTextRetrieval",
    ],
    "train_test_leakage": [
        "AVEDatasetClassification",
        "AVEDatasetVideoClassification",
        "AfriHateClassification",
        "AfriSentiClassification",
        "AllegroReviews",
        "AmazonCounterfactualClassification",
        "AmazonPolarityVNClassification",
        "AmazonReviewsClassification",
        "AmazonReviewsVNClassification",
        "ArxivClassification",
        "AudioSet",
        "Banking77VNClassification",
        "BengaliDocumentClassification",
        "BrazilianToxicTweetsClassification",
        "CBD",
        "CEDRClassification",
        "CSFDCZMovieReviewSentimentClassification",
        "CSFDSKMovieReviewSentimentClassification",
        "CataloniaTweetClassification",
        "CzechProductReviewSentimentClassification",
        "CzechSoMeSentimentClassification",
        "DKHateClassification",
        "DalajClassification",
        "Ddisco",
        "DeepSentiPers",
        "DutchGovernmentBiasClassification",
        "DutchNewsArticlesClassification",
        "EmitClassification",
        "EmotionAnalysisPlus",
        "EmotionClassification",
        "EmotionVNClassification",
        "EstonianValenceClassification",
        "FilipinoHateSpeechClassification",
        "FinToxicityClassification",
        "FrenkEnClassification",
        "FrenkHrClassification",
        "FrenkSlClassification",
        "GujaratiNewsClassification",
        "HMDB51Classification",
        "HUMEToxicConversationsClassification",
        "HebrewSentimentAnalysis",
        "HinDialectClassification",
        "IFlyTek",
        "IFlyTek.v2",
        "ImdbClassification",
        "ImdbVNClassification",
        "IndicLangClassification",
        "IndicNLPNewsClassification",
        "InjongoIntent",
        "JDReview",
        "JapaneseSentimentClassification",
        "JavaneseIMDBClassification",
        "KinNewsClassification",
        "KorHateSpeechMLClassification",
        "KurdishSentimentClassification",
        "LanguageClassification",
        "MAUDLegalBenchClassification",
        "MTOPDomainClassification",
        "MTOPDomainVNClassification",
        "MTOPIntentClassification",
        "MTOPIntentVNClassification",
        "MacedonianTweetSentimentClassification",
        "MalayalamNewsClassification",
        "MarathiNewsClassification",
        "MasakhaNEWSClassification",
        "MassiveIntentClassification",
        "MassiveIntentVNClassification",
        "MassiveScenarioClassification",
        "MassiveScenarioVNClassification",
        "Moroco",
        "MovieReviewSentimentClassification",
        "MovieReviewSentimentClassification.v2",
        "MultiHateClassification",
        "MultilingualSentiment",
        "MultilingualSentiment.v2",
        "MultilingualSentimentClassification",
        "NLPTwitterAnalysisClassification",
        "NaijaSenti",
        "NoRecClassification",
        "NordicLangClassification",
        "NorwegianParliamentClassification",
        "NorwegianParliamentClassification.v2",
        "NusaParagraphEmotionClassification",
        "NusaParagraphTopicClassification",
        "OPP115DataSecurityLegalBenchClassification",
        "OPP115DoNotTrackLegalBenchClassification",
        "OPP115UserChoiceControlLegalBenchClassification",
        "OdiaNewsClassification",
        "OpenTenderClassification",
        "PatentClassification",
        "PerShopDomainClassification",
        "PerShopIntentClassification",
        "PersianTextEmotion",
        "RomanianReviewsSentiment",
        "RomanianSentimentClassification",
        "RuNLUIntentClassification",
        "RuSciBenchCoreRiscClassification",
        "RuSciBenchGRNTIClassification.v2",
        "SDSEyeProtectionClassification",
        "SDSGlovesClassification",
        "SIB200Classification",
        "SIB200Classification.v2",
        "SIDClassification",
        "ScandiSentClassification",
        "SentiRuEval2016",
        "SentimentDKSF",
        "SlovakHateSpeechClassification",
        "SlovakMovieReviewSentimentClassification",
        "SouthAfricanLangClassification",
        "SwedishSentimentClassification",
        "SwedishSentimentClassification.v2",
        "SwissJudgementClassification",
        "SynPerTextToneClassification",
        "SynPerTextToneClassification.v3",
        "TNews",
        "TNews.v2",
        "TamilNewsClassification",
        "TeluguAndhraJyotiNewsClassification",
        "TenKGnadClassification",
        "ToxicChatClassification",
        "ToxicConversationsClassification",
        "ToxicConversationsVNClassification",
        "TweetSarcasmClassification",
        "TweetSentimentExtractionVNClassification",
        "UkrFormalityClassification",
        "VABBMultiLabelClassification",
        "VaccinChatNLClassification",
        "WRIMEClassification",
        "Waimai",
        "WikipediaBioMetChemClassification",
        "WikipediaChemFieldsClassification",
        "WikipediaCrystallographyAnalyticalClassification",
        "WikipediaTheoreticalAppliedClassification",
        "WisesightSentimentClassification",
        "YahooAnswersTopicsClassification",
        "YueOpenriceReviewClassification",
    ],
    "duplicate_image": [
        "AROCocoOrder",
        "AROFlickrOrder",
        "AROVisualAttribution",
        "AROVisualRelation",
        "CIRRIT2IRetrieval",
        "EDIST2ITRetrieval",
        "EncyclopediaVQAIT2ITRetrieval",
        "FER2013",  # documented to contain duplicate/near-duplicate images
        "FER2013ZeroShot",  # documented to contain duplicate/near-duplicate images
        "FORBI2IRetrieval",  # fingerprint corpus contains near-duplicate captures of the same print
        "FashionIQIT2IRetrieval",
        "ImageCoDeT2IRetrieval",
        "InfoSeekIT2ITRetrieval",
        "InfoSeekIT2TRetrieval",
        "LLaVAIT2TRetrieval",
        "MomentSeekerTI2VRetrieval",
        "OVENIT2ITRetrieval",
        "PatchCamelyon",  # adjacent, overlapping WSI patches are inherent to the source data
        "PatchCamelyonZeroShot",  # adjacent, overlapping WSI patches are inherent to the source data
        "RParisEasyI2IRetrieval",
        "RParisHardI2IRetrieval",
        "RParisMediumI2IRetrieval",
        "ReMuQIT2TRetrieval",
        "SOPI2IRetrieval",
        "SoundingEarthA2IRetrieval",
        "SoundingEarthI2ARetrieval",
        "SugarCrepe",
        "VQA2IT2TRetrieval",
        "WebQAT2ITRetrieval",
        "XFlickr30kCoT2IRetrieval",
        "XM3600T2IRetrieval",
    ],
    "duplicate_pairs": [
        "BibleNLPBitextMining",
        "CREMADPairClassification",  # pairs constructed combinatorially from a small pool of audio clips
        "DiaBlaBitextMining",
        "ESC50PairClassification",  # pairs constructed combinatorially from a small pool of audio clips
        "FalseFriendsGermanEnglish",
        "LinceMTBitextMining",
        "ParsinluEntail",
        "Query2Query",
        "RUParaPhraserSTS",
        "SICK-R-VN",
        "SICKFr",
        "STS12",
        "STS14",
        "STS22",
        "STS22.v2",
        "TwitterURLCorpus",
        "TwitterURLCorpus-VN",
        "VocalSoundPairClassification",  # pairs constructed combinatorially from a small pool of audio clips
        "VoxPopuliAccentPairClassification",  # pairs constructed combinatorially from a small pool of audio clips
    ],
    "duplicate_audio": [
        "AmbientAcousticContext",
        "AmbientAcousticContextClustering",
        "CLDAT2ARetrieval",
        "FSD2019Kaggle",
        "GTZANGenre",  # repeated short clips sampled from the same tracks
        "GTZANGenreClustering",  # repeated short clips sampled from the same tracks
        "Kinetics400VA",  # multiple clips can share the same soundtrack/source video
        "Kinetics400VAZeroShot",
        "Kinetics600VA",
        "Kinetics600VAZeroShot",
        "Kinetics700VA",
        "Kinetics700VAZeroShot",
        "NSynth",  # repeated notes across instrument/pitch/velocity combinations
        "SpeechCommands",  # many repeated recordings of the same short command word
        "SpeechCommandsZeroshotv0.01",
        "SpeechCommandsZeroshotv0.02",
        "WorldSenseAudioVideoClassification",  # multiple QA rows share the same underlying video/audio
        "WorldSenseAudioVideoZeroShot",
    ],
    "duplicate_video": [
        "MMVUVideoCentricQA",
        "MomentSeekerTV2VRetrieval",
        "WorldSenseAudioVideoClassification",  # multiple QA rows share the same underlying video/audio
        "WorldSenseAudioVideoZeroShot",
        "WorldSenseVideoClassification",  # multiple QA rows share the same underlying video/audio
        "WorldSenseVideoZeroShot",
    ],
    "relevant_docs_exceed_corpus": [
        "MSMARCO-FaHardNegatives",
        "MSMARCOHardNegatives",
        "NeuCLIR2022RetrievalHardNegatives",
        "NeuCLIR2023RetrievalHardNegatives",
    ],
    "small_image": [
        "FORBI2IRetrieval",  # min 1x1px query image present
        "GLDv2I2IRetrieval",  # min 7x7px document image present
        "Imagenet1k",  # min 8x10px image present
    ],
    "zero_relevant_docs": [
        "BrightRetrieval",
        "TwitterHjerneRetrieval",
    ],
    "impossible_unique_count": [
        # ImageTextPairClassification / vision-language compositionality
        # benchmarks: text_statistics/image_statistics is computed over
        # several candidates (distractors/permutations) per example, so
        # num_samples (example count) isn't the right denominator -- the
        # stats format has no field for "candidates per example" to fix this
        # properly, unlike Retrieval's num_documents/num_queries.
        "AROCocoOrder",  # multiple caption-order permutations per example
        "AROFlickrOrder",  # multiple caption-order permutations per example
        "AROVisualAttribution",  # correct + distractor attribution captions per example
        "AROVisualRelation",  # correct + distractor relation captions per example
        "ImageCoDe",  # multiple candidate images per example
        "SugarCrepe",  # correct + hard-negative captions per example
        "Winoground",  # exactly 2 texts and 2 images per example (400 samples -> 800 of each)
        # Genuine stats-generation inconsistency: unique_texts exceeds the
        # corpus/query count by a large, non-integer-multiple factor, not
        # explained by any known per-example packing -- num_documents/
        # num_queries themselves look undercounted relative to the actual
        # text rows scanned when the stats were computed.
        "BUCC",  # sentence2_statistics unique count is ~32x num_samples
        "MSMARCOv2",  # documents_text_statistics unique count is ~9x num_documents
        "MrTidyRetrieval",  # documents_text_statistics unique count exceeds num_documents by ~13%
        "MuPLeR-retrieval",  # both documents_ and queries_text_statistics exceed their counts
        "R2MEDBioinformaticsRetrieval",
        "R2MEDMedQADiagRetrieval",
        "R2MEDMedicalSciencesRetrieval",
        "R2MEDPMCTreatmentRetrieval",  # queries_text_statistics: 150 unique vs 5 queries
    ],
}


def _task_known_issues() -> dict[str, set[str]]:
    task_known_issues: dict[str, set[str]] = {}
    for kind, names in KNOWN_ISSUES.items():
        for name in names:
            task_known_issues.setdefault(name, set()).add(kind)
    return task_known_issues


_DUPLICATE_RATIO_TOLERANCE = 0.01
# Large corpora routinely contain a small fraction of naturally-repeated
# short items (numbers, single words, boilerplate phrases, common audio
# clips reused across pairs, ...) without that being a quality problem, so
# an exact `unique_count == expected_count` requirement is too strict; the
# duplicate check only fires once the *share* of duplicates exceeds this.

# Values strictly greater than this many pixels (on both width and height)
# are required; e.g. 8 means an 8px-wide image is flagged as too small.
_MIN_IMAGE_DIMENSION = 8

_LENGTH_OUTLIER_RATIO = 50
_DURATION_OUTLIER_RATIO = 50
_DIMENSION_OUTLIER_RATIO = 20

# Outlier checks are reported as warnings, not failures
_WARNING_CHECK_KINDS = {"long_text", "long_audio", "long_video", "large_video_frame"}

assert not (set(KNOWN_ISSUES) & _WARNING_CHECK_KINDS), (
    "A warning-kind check must not appear in KNOWN_ISSUES (it can never fail "
    "the test, so listing it would make the stale-entry check permanently fire)."
)

_AV_DURATION_RELATIVE_TOLERANCE = 0.05

_DUPLICATE_CHECK_EXEMPT_SUFFIXES = ("1_statistics", "2_statistics")
# Fields where `expected_count` (== num_samples) is *known* to be the wrong
# denominator by construction -- several summaries/labels are packed into
# each row, so `unique_texts` naturally exceeding num_samples several-fold
# isn't a stats bug, just a field this test has no accurate count for.
_DUPLICATE_CHECK_EXEMPT_FIELDS = {
    "candidates_labels_text_statistics",
    "human_summaries_statistics",
    "machine_summaries_statistics",
}


def _is_duplicate_check_exempt(field: str) -> bool:
    return (
        field.endswith(_DUPLICATE_CHECK_EXEMPT_SUFFIXES)
        or field in _DUPLICATE_CHECK_EXEMPT_FIELDS
    )


def _is_impossible_count_exempt(field: str) -> bool:
    return field in _DUPLICATE_CHECK_EXEMPT_FIELDS


def _expected_unique_count(
    field: str,
    num_samples: int | None,
    num_queries: int | None,
    num_documents: int | None,
) -> int | None:
    """The row count a field's `unique_*` should be compared against.

    For Retrieval, `num_samples` is `num_queries + num_documents` combined,
    not "one row per unique value" for either `documents_*` or `queries_*`
    fields individually -- those need their own, correct denominator.
    """
    if field.startswith("documents_"):
        return num_documents
    if field.startswith("queries_"):
        return num_queries
    return num_samples


def _is_duplicated(num_samples: int, unique_count: int) -> bool:
    """True once the duplicate share exceeds `_DUPLICATE_RATIO_TOLERANCE`."""
    return unique_count < num_samples * (1 - _DUPLICATE_RATIO_TOLERANCE)


def _is_outlier(average: float | None, value: float | None, ratio: float) -> bool:
    """True once `value` exceeds `average` by more than `ratio`x."""
    return (
        average is not None
        and value is not None
        and average > 0
        and value > average * ratio
    )


def _is_below_min(value: float | None, minimum: float) -> bool:
    return value is not None and not (value > minimum)


def _iter_stat_fields(split_stats: SplitDescriptiveStatistics) -> list[tuple[str, Any]]:
    """Top-level (field_name, value) pairs holding nested statistics dicts.

    Skips ``hf_subset_descriptive_stats``, which is handled separately by the
    caller since it holds per-language *splits*, not a statistics dict.
    """
    return [
        (key, value)
        for key, value in split_stats.items()
        if key != "hf_subset_descriptive_stats" and isinstance(value, dict)
    ]


def _text_field_quality(
    name: str, split: str, field: str, stats: TextStatistics
) -> tuple[int, list[tuple[str, str]]]:
    errors: list[tuple[str, str]] = []
    min_text_length = stats["min_text_length"]
    if not (min_text_length > 3):
        errors.append(
            (
                f"short_text:{field}",
                f"{name} ({split}) contains documents which are extremely short in {field} ({min_text_length=}), this likely indicate poor quality samples.",
            )
        )

    max_text_length = stats["max_text_length"]
    average_text_length = stats["average_text_length"]
    if _is_outlier(average_text_length, max_text_length, _LENGTH_OUTLIER_RATIO):
        errors.append(
            (
                f"long_text:{field}",
                f"{name} ({split}) contains a document far longer than the rest of {field} ({max_text_length=}, {average_text_length=}), this can indicate a truncation/concatenation bug.",
            )
        )

    return stats["unique_texts"], errors


def _image_field_quality(
    name: str, split: str, field: str, stats: ImageStatistics
) -> tuple[int, list[tuple[str, str]]]:
    errors: list[tuple[str, str]] = []
    min_image_width = stats["min_image_width"]
    min_image_height = stats["min_image_height"]
    if not (min_image_width > _MIN_IMAGE_DIMENSION) or not (
        min_image_height > _MIN_IMAGE_DIMENSION
    ):
        errors.append(
            (
                f"small_image:{field}",
                f"{name} ({split}) contains images which are extremely small in {field} ({min_image_width=}, {min_image_height=}), this likely indicate poor quality samples.",
            )
        )
    return stats["unique_images"], errors


def _audio_field_quality(
    name: str, split: str, field: str, stats: AudioStatistics
) -> tuple[int, list[tuple[str, str]]]:
    errors: list[tuple[str, str]] = []

    min_duration_seconds = stats["min_duration_seconds"]
    if not (min_duration_seconds > 0):
        errors.append(
            (
                f"zero_length_audio:{field}",
                f"{name} ({split}) has zero-length audio clips in {field} ({min_duration_seconds=})",
            )
        )

    max_duration_seconds = stats["max_duration_seconds"]
    average_duration_seconds = stats["average_duration_seconds"]
    if _is_outlier(
        average_duration_seconds, max_duration_seconds, _DURATION_OUTLIER_RATIO
    ):
        errors.append(
            (
                f"long_audio:{field}",
                f"{name} ({split}) has an audio clip far longer than the rest of {field} ({max_duration_seconds=}, {average_duration_seconds=}), this can indicate a wrong clip boundary.",
            )
        )

    return stats["unique_audios"], errors


def _video_field_quality(
    name: str, split: str, field: str, stats: VideoStatistics
) -> tuple[int, list[tuple[str, str]]]:
    errors: list[tuple[str, str]] = []

    min_duration_seconds = stats.get("min_duration_seconds")
    if min_duration_seconds is not None and not (min_duration_seconds > 0):
        errors.append(
            (
                f"zero_length_video:{field}",
                f"{name} ({split}) has zero-length video clips in {field} ({min_duration_seconds=})",
            )
        )

    min_width = stats.get("min_width")
    min_height = stats.get("min_height")
    if _is_below_min(min_width, _MIN_IMAGE_DIMENSION) or _is_below_min(
        min_height, _MIN_IMAGE_DIMENSION
    ):
        errors.append(
            (
                f"small_video:{field}",
                f"{name} ({split}) contains video frames which are extremely small in {field} ({min_width=}, {min_height=}), this likely indicate poor quality samples.",
            )
        )

    max_duration_seconds = stats.get("max_duration_seconds")
    average_duration_seconds = stats.get("average_duration_seconds")
    if _is_outlier(
        average_duration_seconds, max_duration_seconds, _DURATION_OUTLIER_RATIO
    ):
        errors.append(
            (
                f"long_video:{field}",
                f"{name} ({split}) has a video clip far longer than the rest of {field} ({max_duration_seconds=}, {average_duration_seconds=}), this can indicate a wrong clip boundary.",
            )
        )

    max_width = stats.get("max_width")
    average_width = stats.get("average_width")
    max_height = stats.get("max_height")
    average_height = stats.get("average_height")
    if _is_outlier(average_width, max_width, _DIMENSION_OUTLIER_RATIO) or _is_outlier(
        average_height, max_height, _DIMENSION_OUTLIER_RATIO
    ):
        errors.append(
            (
                f"large_video_frame:{field}",
                f"{name} ({split}) has a video frame far larger than the rest of {field} ({max_width=}, {average_width=}, {max_height=}, {average_height=}), this can indicate a mis-decoded frame.",
            )
        )

    return stats["unique_videos"], errors


def _label_field_quality(
    name: str, split: str, field: str, stats: LabelStatistics
) -> list[tuple[str, str]]:
    errors: list[tuple[str, str]] = []
    unique_labels = stats["unique_labels"]
    if unique_labels < 2:
        errors.append(
            (
                f"too_few_labels:{field}",
                f"{name} ({split}) has fewer than 2 unique labels in {field} ({unique_labels=})",
            )
        )
    return errors


def _label_or_score_field_quality(
    name: str, split: str, field: str, stats: dict[str, Any]
) -> list[tuple[str, str]]:
    """Checks the remaining (non-media, non-duplicate-eligible) shapes."""
    if "unique_labels" in stats:
        return _label_field_quality(name, split, field, cast("LabelStatistics", stats))

    if "min_score" in stats and "max_score" in stats:
        score_stats = cast("ScoreStatistics", stats)
        if score_stats["min_score"] == score_stats["max_score"]:
            return [
                (
                    f"zero_label_variance:{field}",
                    f"{name} ({split}) has zero label variance in {field} (min_score == max_score == {score_stats['min_score']})",
                )
            ]

    elif "min_relevant_docs_per_query" in stats:
        relevant_docs_stats = cast("RelevantDocsStatistics", stats)
        if relevant_docs_stats["min_relevant_docs_per_query"] == 0:
            return [
                (
                    f"zero_relevant_docs:{field}",
                    f"{name} ({split}) has queries with zero relevant documents in {field}",
                )
            ]

    elif "min_top_ranked_per_query" in stats:
        top_ranked_stats = cast("TopRankedStatistics", stats)
        if top_ranked_stats["min_top_ranked_per_query"] == 0:
            return [
                (
                    f"zero_top_ranked:{field}",
                    f"{name} ({split}) has queries with zero top-ranked candidates in {field}",
                )
            ]

    return []


def _field_quality(
    name: str,
    split: str,
    field: str,
    stats: dict[str, Any],
    expected_count: int | None,
) -> list[tuple[str, str]]:
    unique_count: int | None = None
    modality: str | None = None

    if "unique_texts" in stats:
        modality = "text"
        unique_count, errors = _text_field_quality(
            name, split, field, cast("TextStatistics", stats)
        )
    elif "unique_images" in stats:
        modality = "image"
        unique_count, errors = _image_field_quality(
            name, split, field, cast("ImageStatistics", stats)
        )
    elif "unique_audios" in stats:
        modality = "audio"
        unique_count, errors = _audio_field_quality(
            name, split, field, cast("AudioStatistics", stats)
        )
    elif "unique_videos" in stats:
        modality = "video"
        unique_count, errors = _video_field_quality(
            name, split, field, cast("VideoStatistics", stats)
        )
    else:
        errors = _label_or_score_field_quality(name, split, field, stats)

    if (
        unique_count is not None
        and expected_count is not None
        and not _is_duplicate_check_exempt(field)
        and _is_duplicated(expected_count, unique_count)
    ):
        errors.append(
            (
                f"duplicate_{modality}:{field}",
                f"{name} ({split}) contains {modality} duplicates in {field} ({expected_count=}, unique_{modality}s={unique_count})",
            )
        )
    elif (
        unique_count is not None
        and expected_count is not None
        and not _is_impossible_count_exempt(field)
        and unique_count > expected_count
    ):
        errors.append(
            (
                f"impossible_unique_count:{field}",
                f"{name} ({split}) has more unique {modality}s in {field} than possible ({expected_count=}, unique_{modality}s={unique_count}) -- the stats are internally inconsistent.",
            )
        )

    return errors


def _relevant_docs_bound_quality(
    name: str, split: str, split_stats: SplitDescriptiveStatistics
) -> list[tuple[str, str]]:
    """Relevant docs referenced in qrels must be a subset of the corpus."""
    relevant_docs_stats = split_stats.get("relevant_docs_statistics")
    num_documents = split_stats.get("num_documents")
    if not isinstance(relevant_docs_stats, dict) or num_documents is None:
        return []

    unique_relevant_docs = relevant_docs_stats.get("unique_relevant_docs")
    if unique_relevant_docs is not None and unique_relevant_docs > num_documents:
        return [
            (
                "relevant_docs_exceed_corpus",
                f"{name} ({split}) has more unique relevant docs than documents in the corpus ({unique_relevant_docs=}, {num_documents=}), qrels likely reference IDs missing from the corpus.",
            )
        ]
    return []


def _audio_video_pair_quality(
    name: str, split: str, split_stats: SplitDescriptiveStatistics
) -> list[tuple[str, str]]:
    errors: list[tuple[str, str]] = []
    fields = dict(_iter_stat_fields(split_stats))
    for field, stats in fields.items():
        if "unique_audios" not in stats:
            continue
        video_field = field.replace("audio", "video")
        video_stats = fields.get(video_field)
        if not isinstance(video_stats, dict) or "unique_videos" not in video_stats:
            continue

        audio_value = stats.get("average_duration_seconds")
        video_value = video_stats.get("average_duration_seconds")
        if audio_value is None or video_value is None:
            continue
        if not math.isclose(
            audio_value, video_value, rel_tol=_AV_DURATION_RELATIVE_TOLERANCE
        ):
            errors.append(
                (
                    f"audio_video_length_mismatch:{field}",
                    f"{name} ({split}) has mismatched average_duration_seconds between {field} ({audio_value}) and {video_field} ({video_value})",
                )
            )

    return errors


def _split_quality(
    name: str, split: str, split_stats: SplitDescriptiveStatistics
) -> list[tuple[str, str]]:
    """Runs all checks against a single split's statistics"""
    errors: list[tuple[str, str]] = []

    num_samples = cast("int | None", split_stats.get("num_samples"))
    if num_samples is not None and num_samples == 0:
        errors.append(
            ("empty_split", f"{name} ({split}) has no samples ({num_samples=})")
        )

    unique_pairs = cast("int | None", split_stats.get("unique_pairs"))
    if unique_pairs is not None and num_samples is not None:
        if _is_duplicated(num_samples, unique_pairs):
            errors.append(
                (
                    "duplicate_pairs",
                    f"{name} ({split}) contains duplicate pairs ({num_samples=}, {unique_pairs=})",
                )
            )
        elif unique_pairs > num_samples:
            errors.append(
                (
                    "impossible_unique_count",
                    f"{name} ({split}) has more unique pairs than possible ({num_samples=}, {unique_pairs=}) -- the stats are internally inconsistent.",
                )
            )

    num_queries = cast("int | None", split_stats.get("num_queries"))
    num_documents = cast("int | None", split_stats.get("num_documents"))
    for field, stats in _iter_stat_fields(split_stats):
        expected_count = _expected_unique_count(
            field, num_samples, num_queries, num_documents
        )
        errors += _field_quality(name, split, field, stats, expected_count)

    errors += _relevant_docs_bound_quality(name, split, split_stats)
    errors += _audio_video_pair_quality(name, split, split_stats)

    # train-test leakage
    samples_in_train = split_stats.get("samples_in_train", None)
    if not (samples_in_train is None or samples_in_train == 0):
        errors.append(
            (
                "train_test_leakage",
                f"{name} ({split}) has an overlap between train and test ({samples_in_train=})",
            )
        )
    return errors


def _task_quality(task: AbsTask) -> list[tuple[str, str]]:
    """Returns ``(check_id, message)`` pairs; see `_split_quality`."""
    desc_stats = task.metadata.descriptive_stats
    name = task.metadata.name

    if desc_stats is None:
        return []

    findings: list[tuple[str, str]] = []
    for split_name, split_stats in desc_stats.items():
        findings += _split_quality(name, split_name, split_stats)

        hf_subset_stats = split_stats.get("hf_subset_descriptive_stats")
        if not hf_subset_stats:
            continue

        subsets_by_check: dict[str, list[str]] = {}
        for hf_subset, subset_stats in hf_subset_stats.items():
            for check_id, _ in _split_quality(
                f"{name} [{hf_subset}]", split_name, subset_stats
            ):
                subsets_by_check.setdefault(check_id, []).append(hf_subset)

        for check_id, subsets in subsets_by_check.items():
            shown = ", ".join(subsets[:5])
            more = f", +{len(subsets) - 5} more" if len(subsets) > 5 else ""
            findings.append(
                (
                    check_id,
                    f"{name} ({split_name}) fails '{check_id}' in {len(subsets)} hf_subset(s): {shown}{more}",
                )
            )

    return findings


def test_dataset_quality() -> None:
    tasks = get_tasks(
        exclude_superseded=False, exclude_aggregate=True, exclude_beta=False
    )
    task_known_issues = _task_known_issues()

    errors: list[str] = []
    findings: list[str] = []
    stale_known_issues: list[str] = []
    for task in tasks:
        name = task.metadata.name
        exempt_check_ids = task_known_issues.get(name, set())
        seen_check_ids: set[str] = set()

        for check_id, message in _task_quality(task):
            kind = check_id.split(":", 1)[0]
            seen_check_ids.add(kind)
            if kind in _WARNING_CHECK_KINDS:
                findings.append(message)
                continue

            if kind not in exempt_check_ids:
                errors.append(message)

        for stale_check_id in exempt_check_ids - seen_check_ids:
            stale_known_issues.append(
                f"{name} is listed in KNOWN_ISSUES[{stale_check_id!r}] but no longer fails that check "
                "-- the dataset/stats were likely fixed; remove this entry from KNOWN_ISSUES."
            )

    for message in findings:
        warnings.warn(message, stacklevel=2)

    errors = stale_known_issues + errors

    if errors:
        raise AssertionError("\n".join([str(e) for e in errors]))
