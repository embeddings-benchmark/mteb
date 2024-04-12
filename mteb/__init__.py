from __future__ import annotations

from importlib.metadata import version

from mteb.evaluation import *

__version__ = version("mteb")  # fetch version from install metadata


MTEB_MAIN_EN = [
    "AmazonCounterfactualClassification",
    "AmazonPolarityClassification",
    "AmazonReviewsClassification",
    "ArguAna",
    "ArxivClusteringP2P",
    "ArxivClusteringS2S",
    "AskUbuntuDupQuestions",
    "BIOSSES",
    "Banking77Classification",
    "BiorxivClusteringP2P",
    "BiorxivClusteringS2S",
    "CQADupstackAndroidRetrieval",
    "CQADupstackEnglishRetrieval",
    "CQADupstackGamingRetrieval",
    "CQADupstackGisRetrieval",
    "CQADupstackMathematicaRetrieval",
    "CQADupstackPhysicsRetrieval",
    "CQADupstackProgrammersRetrieval",
    "CQADupstackStatsRetrieval",
    "CQADupstackTexRetrieval",
    "CQADupstackUnixRetrieval",
    "CQADupstackWebmastersRetrieval",
    "CQADupstackWordpressRetrieval",
    "ClimateFEVER",
    "DBPedia",
    "EmotionClassification",
    "FEVER",
    "FiQA2018",
    "HotpotQA",
    "ImdbClassification",
    "MSMARCO",
    "MTOPDomainClassification",
    "MTOPIntentClassification",
    "MassiveIntentClassification",
    "MassiveScenarioClassification",
    "MedrxivClusteringP2P",
    "MedrxivClusteringS2S",
    "MindSmallReranking",
    "NFCorpus",
    "NQ",
    "QuoraRetrieval",
    "RedditClustering",
    "RedditClusteringP2P",
    "SCIDOCS",
    "SICK-R",
    "STS12",
    "STS13",
    "STS14",
    "STS15",
    "STS16",
    "STS17",
    "STS22",
    "STSBenchmark",
    "SciDocsRR",
    "SciFact",
    "SprintDuplicateQuestions",
    "StackExchangeClustering",
    "StackExchangeClusteringP2P",
    "StackOverflowDupQuestions",
    "SummEval",
    "TRECCOVID",
    "Touche2020",
    "ToxicConversationsClassification",
    "TweetSentimentExtractionClassification",
    "TwentyNewsgroupsClustering",
    "TwitterSemEval2015",
    "TwitterURLCorpus",
]


MTEB_INSTRUCTION_FOLLOWING = [
    "RobustInstructionRetrieval",
    "NewsInstructionRetrieval",
    "CoreInstructionRetrieval",
]

  
MTEB_RETRIEVAL_LAW = [
    "LegalSummarization",
    "LegalBenchConsumerContractsQA",
    "LegalBenchCorporateLobbying",
    "AILACasedocs",
    "AILAStatutes",
    "LeCaRDv2",
    "LegalQuAD",
    "GerDaLIRSmall",
]
