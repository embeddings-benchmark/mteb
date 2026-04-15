from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import torch
from sentence_transformers import SentenceTransformer

from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.types import PromptType

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput

BIDIRLM_OMNI_LANGUAGES = [
    "afr-Latn",
    "amh-Ethi",
    "ara-Arab",
    "aze-Latn",
    "bel-Cyrl",
    "bul-Cyrl",
    "ben-Beng",
    "bos-Latn",
    "cat-Latn",
    "ceb-Latn",
    "ces-Latn",
    "cym-Latn",
    "dan-Latn",
    "deu-Latn",
    "ell-Grek",
    "eng-Latn",
    "spa-Latn",
    "est-Latn",
    "eus-Latn",
    "fas-Arab",
    "fin-Latn",
    "fra-Latn",
    "gle-Latn",
    "glg-Latn",
    "guj-Gujr",
    "hau-Latn",
    "heb-Hebr",
    "hin-Deva",
    "hrv-Latn",
    "hat-Latn",
    "hun-Latn",
    "hye-Armn",
    "ind-Latn",
    "ibo-Latn",
    "isl-Latn",
    "ita-Latn",
    "jpn-Jpan",
    "jav-Latn",
    "kat-Geor",
    "kaz-Cyrl",
    "kan-Knda",
    "kor-Hang",
    "kir-Cyrl",
    "lit-Latn",
    "lav-Latn",
    "mlg-Latn",
    "mkd-Cyrl",
    "mal-Mlym",
    "mar-Deva",
    "msa-Latn",
    "mlt-Latn",
    "mya-Mymr",
    "nob-Latn",
    "nep-Deva",
    "nld-Latn",
    "nso-Latn",
    "nya-Latn",
    "pan-Guru",
    "pol-Latn",
    "pus-Arab",
    "por-Latn",
    "ron-Latn",
    "rus-Cyrl",
    "snd-Arab",
    "sin-Sinh",
    "slk-Latn",
    "slv-Latn",
    "sna-Latn",
    "som-Latn",
    "sqi-Latn",
    "srp-Cyrl",
    "sun-Latn",
    "swe-Latn",
    "swa-Latn",
    "tam-Taml",
    "tel-Telu",
    "tha-Thai",
    "tgl-Latn",
    "tur-Latn",
    "ukr-Cyrl",
    "urd-Arab",
    "vie-Latn",
    "wol-Latn",
    "xho-Latn",
    "yor-Latn",
    "cmn-Hans",
    "zul-Latn",
]

BIDIRLM_OMNI_CITATION = """@misc{boizard2026bidirlmtextomnimodalbidirectional,
      title={BidirLM: From Text to Omnimodal Bidirectional Encoders by Adapting and Composing Causal LLMs},
      author={Nicolas Boizard and Théo Deschamps-Berger and Hippolyte Gisserot-Boukhlef and Céline Hudelot and Pierre Colombo},
      year={2026},
      eprint={2604.02045},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2604.02045},
}"""

BIDIRLM_OMNI_TRAINING_DATASETS = {
    "AFQMC",
    "OPUS-100",
    "AdvertiseGen",
    "CAIL2019-SCM",
    "CHEF",
    "CINLID",
    "ChatMed_Consult_Dataset",
    "ChineseSTS",
    "CodeFeedback",
    "ELI5_custom",
    "EmotionClassification",
    "Expertqa",
    "FollowIR",
    "GooAQ",
    "InF-IR",
    "JW300",
    "LAION-Audio-300M",
    "LCSTS",
    "MS_COCO",
    "MEDI2BGE",
    "MIRACL",
    "MAmmoTH2",
    "MSMARCO",
    "Multi-CPR",
    "NaturalQuestions",
    "NFCorpus",
    "OpenOrca",
    "PAQ",
    "PubMedQA",
    "QBQTC",
    "RefGPT",
    "SearchQA",
    "SimCLUE",
    "SQuAD",
    "SyntheticClassificationData",
    "T2Ranking",
    "THUCNews",
    "TED Talks",
    "TriviaQA",
    "UMETRIP-QA",
    "WebCPM",
    "WikiAnswers",
    "WikiMatrix",
    "atec",
    "bq",
    "cCOVID-News",
    "cMedQA-V2.0",
    "ccnews",
    "cmnli",
    "cmrc2018",
    "colpali_train_set",
    "contract-nli",
    "csl",
    "dureader",
    "dureader_mrc",
    "esci",
    "law-gpt",
    "lawzhidao",
    "librispeech_asr",
    "lima-chinese",
    "mldr",
    "mmarco-chinese",
    "mnli",
    "natcap",
    "nli_zh",
    "nllb",
    "ocnli",
    "rag-dataset-12000",
    "simcse_sup_nli",
    "webgpt_comparisons",
    "webqa",
    "wikipedia-nq",
    "xnli_zh",
    "yahoo-answers",
    "DRCD",
}

_TASK_PROMPTS: dict[str, str | dict[str, str]] = {
    # MTEB tasks
    "AmazonCounterfactualClassification": "Given an Amazon review, judge whether it is counterfactual.",
    "AmazonPolarityClassification": "Classifying Amazon reviews into positive or negative sentiment",
    "AmazonReviewsClassification": "Classifying the given Amazon review into its appropriate rating category",
    "Banking77Classification": "Given an online banking query, find the corresponding intents",
    "EmotionClassification": "Classify the emotion expressed in the given Twitter message into one of the six emotions: anger, fear, joy, love, sadness, and surprise",
    "ImdbClassification": "Classifying the sentiment expressed in the given movie review text from the IMDB dataset",
    "MassiveIntentClassification": "Given a user utterance as query, find the user intents",
    "MassiveScenarioClassification": "Given a user utterance as query, find the user scenarios",
    "MTOPDomainClassification": "Classifying the intent domain of the given utterance in task-oriented conversation",
    "MTOPIntentClassification": "Classifying the intent of the given utterance in task-oriented conversation",
    "ToxicConversationsClassification": "Classifying the given comments as either toxic or not toxic",
    "TweetSentimentExtractionClassification": "Classifying the sentiment of a given tweet as either positive, negative, or neutral",
    "TNews": "Categorizing the given news title",
    "IFlyTek": "Given an App description text, find the appropriate fine-grained category",
    "MultilingualSentiment": "Classifying sentiment of the customer review into positive, neutral, or negative",
    "JDReview": "Classifying sentiment of the customer review for iPhoneinto positive or negative",
    "OnlineShopping": "Classifying sentiment of the customer reviewinto positive or negative",
    "Waimai": "Classify the customer review from a food takeaway platform into positive or negative",
    "ArxivClusteringP2P": "Identify the main and secondary category of Arxiv papers based on the titles and abstracts",
    "ArxivClusteringS2S": "Identify the main and secondary category of Arxiv papers based on the titles",
    "BiorxivClusteringP2P": "Identify the main category of Biorxiv papers based on the titles and abstracts",
    "BiorxivClusteringS2S": "Identify the main category of Biorxiv papers based on the titles",
    "MedrxivClusteringP2P": "Identify the main category of Medrxiv papers based on the titles and abstracts",
    "MedrxivClusteringS2S": "Identify the main category of Medrxiv papers based on the titles",
    "RedditClustering": "Identify the topic or theme of Reddit posts based on the titles",
    "RedditClusteringP2P": "Identify the topic or theme of Reddit posts based on the titles and posts",
    "StackExchangeClustering": "Identify the topic or theme of StackExchange posts based on the titles",
    "StackExchangeClusteringP2P": "Identify the topic or theme of StackExchange posts based on the given paragraphs",
    "TwentyNewsgroupsClustering": "Identify the topic or theme of the given news articles",
    "CLSClusteringS2S": "Identify the main category of scholar papers based on the titles",
    "CLSClusteringP2P": "Identify the main category of scholar papers based on the titles and abstracts",
    "ThuNewsClusteringS2S": "Identify the topic or theme of the given news articles based on the titles",
    "ThuNewsClusteringP2P": "Identify the topic or theme of the given news articles based on the titles and contents",
    "AskUbuntuDupQuestions": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question",
    "MindSmallReranking": "Given a query, retrieve documents that answer the query.",
    "SciDocsRR": "Given a query, retrieve documents that answer the query.",
    "StackOverflowDupQuestions": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question",
    "SprintDuplicateQuestions": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question",
    "TwitterSemEval2015": "Retrieve semantically similar text.",
    "TwitterURLCorpus": "Retrieve semantically similar text.",
    "T2Reranking": "Given a query, retrieve documents that answer the query.",
    "MmarcoReranking": "Given a query, retrieve documents that answer the query.",
    "CMedQAv1": "Given a query, retrieve documents that answer the query.",
    "CMedQAv2": "Given a query, retrieve documents that answer the query.",
    "Ocnli": "Retrieve semantically similar text.",
    "Cmnli": "Retrieve semantically similar text.",
    "ArguAna": {
        "query": "Given a claim, retrieve documents that support or refute the claim",
        "passage": "Given a claim, retrieve documents that support or refute the claim",
    },
    "ClimateFEVER": "Given a claim, retrieve documents that support or refute the claim",
    "ClimateFEVERHardNegatives": "Given a claim, retrieve documents that support or refute the claim",
    "DBPedia": "Given a query, retrieve documents that answer the query.",
    "FEVER": "Given a claim, retrieve documents that support or refute the claim",
    "FEVERHardNegatives": "Given a claim, retrieve documents that support or refute the claim",
    "FiQA2018": "Given a query, retrieve documents that answer the query.",
    "HotpotQA": "Given a multi-hop question, retrieve documents that can help answer the question",
    "HotpotQAHardNegatives": "Given a multi-hop question, retrieve documents that can help answer the question",
    "MSMARCO": "Given a web search query, retrieve relevant passages that answer the query",
    "NFCorpus": "Given a question, retrieve relevant documents that best answer the question",
    "NQ": "Given a question, retrieve Wikipedia passages that answer the question",
    "QuoraRetrieval": "Given a query, retrieve documents that answer the query.",
    "SCIDOCS": "Given a query, retrieve documents that answer the query.",
    "SciFact": "Given a scientific claim, retrieve documents that support or refute the claim",
    "Touche2020": "Given a query, retrieve documents that answer the query.",
    "Touche2020Retrieval.v3": "Given a query, retrieve documents that answer the query.",
    "TRECCOVID": "Given a query, retrieve documents that answer the query.",
    "T2Retrieval": "Given a question, retrieve passages that answer the question",
    "MMarcoRetrieval": "Given a web search query, retrieve relevant passages that answer the query",
    "DuRetrieval": "Given a question, retrieve passages that answer the question",
    "CovidRetrieval": "Given a query on COVID-19, retrieve documents that answer the query",
    "CmedqaRetrieval": "Given a query, retrieve documents that answer the query.",
    "EcomRetrieval": "Given a query, retrieve documents that answer the query.",
    "MedicalRetrieval": "Given a query, retrieve documents that answer the query.",
    "VideoRetrieval": "Given a query, retrieve documents that answer the query.",
    "STSBenchmarkMultilingualSTS": "Retrieve semantically similar text",
    "SICKFr": "Retrieve semantically similar text",
    "SummEvalFr": "Retrieve semantically similar text.",
    "MasakhaNEWSClassification": "Categorizing the given news title",
    "OpusparcusPC": "Retrieve semantically similar text",
    "PawsX": "Retrieve semantically similar text",
    "AlloProfClusteringP2P": "Identify the main category of scholar papers based on the titles and abstracts",
    "AlloProfClusteringS2S": "Identify the main category of scholar papers based on the titles",
    "HALClusteringS2S": "Identify the main category of scholar papers based on the titles",
    "MasakhaNEWSClusteringP2P": "Identify the topic or theme of the given news articles based on the titles and contents",
    "MasakhaNEWSClusteringS2S": "Identify the topic or theme of the given news articles based on the titles",
    "MLSUMClusteringP2P": "Identify the topic or theme of Reddit posts based on the titles and posts",
    "MLSUMClusteringS2S": "Identify the topic or theme of Reddit posts based on the titles",
    "SyntecReranking": "Given a question, retrieve passages that answer the question",
    "AlloprofReranking": "Given a question, retrieve passages that answer the question",
    "AlloprofRetrieval": "Given a question, retrieve passages that answer the question",
    "BSARDRetrieval": "Given a question, retrieve passages that answer the question",
    "SyntecRetrieval": "Given a question, retrieve passages that answer the question",
    "XPQARetrieval": "Given a question, retrieve passages that answer the question",
    "MintakaRetrieval": "Given a question, retrieve passages that answer the question",
    "CBD": "Classifying the sentiment of a given tweet as either positive, negative, or neutral",
    "PolEmo2.0-IN": "Classifying sentiment of the customer review into positive, neutral, or negative",
    "PolEmo2.0-OUT": "Classifying sentiment of the customer review into positive, neutral, or negative",
    "AllegroReviews": "Classifying sentiment of the customer review into positive, neutral, or negative",
    "PAC": 'Classify the sentence into one of the two types: "BEZPIECZNE_POSTANOWIENIE_UMOWNE" and "KLAUZULA_ABUZYWNA"',
    "SICK-E-PL": "Retrieve semantically similar text",
    "SICK-R-PL": "Retrieve semantically similar text",
    "STS22": "Retrieve semantically similar text",
    "AFQMC": "Retrieve semantically similar text",
    "BQ": "Retrieve semantically similar text",
    "LCQMC": "Retrieve semantically similar text",
    "PAWSX": "Retrieve semantically similar text",
    "QBQTC": "Retrieve semantically similar text",
    "STS12": "Retrieve semantically similar text",
    "PPC": "Retrieve semantically similar text",
    "CDSC-E": "Retrieve semantically similar text",
    "PSC": "Retrieve semantically similar text",
    "8TagsClustering": "Identify the topic or theme of the given news articles",
    "ArguAna-PL": "Given a claim, retrieve documents that support or refute the claim",
    "DBPedia-PL": "Given a query, retrieve documents that answer the query.",
    "FiQA-PL": "Given a query, retrieve documents that answer the query.",
    "HotpotQA-PL": "Given a multi-hop question, retrieve documents that can help answer the question",
    "MSMARCO-PL": "Given a web search query, retrieve relevant passages that answer the query",
    "NFCorpus-PL": "Given a question, retrieve relevant documents that best answer the question",
    "NQ-PL": "Given a question, retrieve Wikipedia passages that answer the question",
    "Quora-PL": "Given a query, retrieve documents that answer the query.",
    "SCIDOCS-PL": "Given a query, retrieve documents that answer the query.",
    "SciFact-PL": "Given a scientific claim, retrieve documents that support or refute the claim",
    "TRECCOVID-PL": "Given a query, retrieve documents that answer the query.",
    "GeoreviewClassification": "Classifying sentiment of the customer review into positive, neutral, or negative",
    "HeadlineClassification": "Categorizing the given news title",
    "InappropriatenessClassification": "Classifying the given comments as either toxic or not toxic",
    "KinopoiskClassification": "Classifying the sentiment expressed in the given movie review text from the IMDB dataset",
    "RuReviewsClassification": "Classifying sentiment of the customer review into positive, neutral, or negative",
    "RuSciBenchGRNTIClassification": "Categorizing the given news title",
    "RuSciBenchOECDClassification": "Categorizing the given news title",
    "GeoreviewClusteringP2P": "Identify the topic or theme of Reddit posts based on the titles and posts",
    "RuSciBenchGRNTIClusteringP2P": "Identify the main category of scholar papers based on the titles and abstracts",
    "RuSciBenchOECDClusteringP2P": "Identify the main category of scholar papers based on the titles and abstracts",
    "TERRa": "Retrieve semantically similar text.",
    "RuBQReranking": "Given a question, retrieve Wikipedia passages that answer the question",
    "RiaNewsRetrieval": "Given a query, retrieve documents that answer the query.",
    "RuBQRetrieval": "Given a question, retrieve Wikipedia passages that answer the question",
    "RUParaPhraserSTS": "Retrieve semantically similar text",
    "RuSTSBenchmarkSTS": "Retrieve semantically similar text",
    "AppsRetrieval": "Given a query, retrieve documents that answer the query.",
    "COIRCodeSearchNetRetrieval": "Given a query, retrieve documents that answer the query.",
    "CodeEditSearchRetrieval": "Given a query, retrieve documents that answer the query.",
    "CodeFeedbackMT": "Given a query, retrieve documents that answer the query.",
    "CodeFeedbackST": "Given a query, retrieve documents that answer the query.",
    "CodeSearchNetCCRetrieval": "Given a query, retrieve documents that answer the query.",
    "CodeSearchNetRetrieval": "Given a query, retrieve documents that answer the query.",
    "CodeTransOceanContest": "Given a query, retrieve documents that answer the query.",
    "CodeTransOceanDL": "Given a query, retrieve documents that answer the query.",
    "CosQA": "Given a query, retrieve documents that answer the query.",
    "StackOverflowQA": "Given a query, retrieve documents that answer the query.",
    "SyntheticText2SQL": "Given a query, retrieve documents that answer the query.",
    "BibleNLPBitextMining": "Retrieve semantically similar text.",
    "BUCC.v2": "Retrieve semantically similar text.",
    "DiaBlaBitextMining": "Retrieve semantically similar text.",
    "FloresBitextMining": "Retrieve semantically similar text.",
    "IN22GenBitextMining": "Retrieve semantically similar text.",
    "IndicGenBenchFloresBitextMining": "Retrieve semantically similar text.",
    "NollySentiBitextMining": "Retrieve semantically similar text.",
    "NTREXBitextMining": "Retrieve semantically similar text.",
    "NusaTranslationBitextMining": "Retrieve semantically similar text.",
    "NusaXBitextMining": "Retrieve semantically similar text.",
    "Tatoeba": "Retrieve semantically similar text.",
    "BulgarianStoreReviewSentimentClassfication": "Classifying sentiment of the customer review into positive, neutral, or negative",
    "CzechProductReviewSentimentClassification": "Classifying sentiment of the customer review into positive, neutral, or negative",
    "GreekLegalCodeClassification": "Categorizing the given news title",
    "DBpediaClassification": "Given an App description text, find the appropriate fine-grained category",
    "FinancialPhrasebankClassification": "Classifying sentiment of the customer review into positive, neutral, or negative",
    "PoemSentimentClassification": "Classifying sentiment of the customer review into positive, neutral, or negative",
    "TweetTopicSingleClassification": "Categorizing the given news title",
    "EstonianValenceClassification": "Classifying sentiment of the customer review into positive, neutral, or negative",
    "FilipinoShopeeReviewsClassification": "Classifying sentiment of the customer review into positive, neutral, or negative",
    "GujaratiNewsClassification": "Categorizing the given news title",
    "SentimentAnalysisHindi": "Classifying sentiment of the customer review into positive, neutral, or negative",
    "IndonesianIdClickbaitClassification": "Categorizing the given news title",
    "ItaCaseholdClassification": "Categorizing the given news title",
    "KorSarcasmClassification": "Classifying sentiment of the customer review into positive, neutral, or negative",
    "KurdishSentimentClassification": "Classifying sentiment of the customer review into positive, neutral, or negative",
    "MacedonianTweetSentimentClassification": "Classifying the sentiment of a given tweet as either positive, negative, or neutral",
    "AfriSentiClassification": "Classifying sentiment of the customer review into positive, neutral, or negative",
    "CataloniaTweetClassification": "Classifying the sentiment of a given tweet as either positive, negative, or neutral",
    "CyrillicTurkicLangClassification": "Given a text, classify its language",
    "IndicLangClassification": "Given a text, classify its language",
    "MultiHateClassification": "Classifying the given comments as either toxic or not toxic",
    "NusaParagraphEmotionClassification": "Classify the emotion expressed in the given Twitter message into one of the six emotions: anger, fear, joy, love, sadness, and surprise",
    "NusaX-senti": "Classifying sentiment of the customer review into positive, neutral, or negative",
    "SwissJudgementClassification": "Classifying sentiment of the customer review into positive, neutral, or negative",
    "NepaliNewsClassification": "Categorizing the given news title",
    "OdiaNewsClassification": "Categorizing the given news title",
    "PunjabiNewsClassification": "Categorizing the given news title",
    "SinhalaNewsClassification": "Categorizing the given news title",
    "CSFDSKMovieReviewSentimentClassification": "Classifying the sentiment expressed in the given movie review text from the IMDB dataset",
    "SiswatiNewsClassification": "Categorizing the given news title",
    "SlovakMovieReviewSentimentClassification": "Classifying the sentiment expressed in the given movie review text from the IMDB dataset",
    "SwahiliNewsClassification": "Categorizing the given news title",
    "TswanaNewsClassification": "Categorizing the given news title",
    "IsiZuluNewsClassification": "Categorizing the given news title",
    "WikiCitiesClustering": "Identify the topic or theme of the given news articles",
    "RomaniBibleClustering": "Identify the topic or theme of the given news articles",
    "ArXivHierarchicalClusteringP2P": "Identify the main and secondary category of Arxiv papers based on the titles and abstracts",
    "ArXivHierarchicalClusteringS2S": "Identify the main and secondary category of Arxiv papers based on the titles",
    "BigPatentClustering.v2": "Identify the main category of scholar papers based on the titles and abstracts",
    "AlloProfClusteringS2S.v2": "Identify the main category of scholar papers based on the titles",
    "HALClusteringS2S.v2": "Identify the main category of scholar papers based on the titles",
    "SIB200ClusteringS2S": "Identify the topic or theme of the given news articles",
    "WikiClusteringP2P.v2": "Identify the topic or theme of the given news articles",
    "PlscClusteringP2P.v2": "Identify the main category of scholar papers based on the titles and abstracts",
    "KorHateSpeechMLClassification": "Classifying the given comments as either toxic or not toxic",
    "MalteseNewsClassification": "Categorizing the given news title",
    "MultiEURLEXMultilabelClassification": "Categorizing the given news title",
    "BrazilianToxicTweetsClassification": "Classifying the given comments as either toxic or not toxic",
    "CTKFactsNLI": "Retrieve semantically similar text",
    "indonli": "Retrieve semantically similar text",
    "ArmenianParaphrasePC": "Retrieve semantically similar text",
    "PawsXPairClassification": "Retrieve semantically similar text",
    "RTE3": "Retrieve semantically similar text",
    "XNLI": "Retrieve semantically similar text",
    "PpcPC": "Retrieve semantically similar text",
    "GermanSTSBenchmark": "Retrieve semantically similar text",
    "SICK-R": "Retrieve semantically similar text",
    "STS13": "Retrieve semantically similar text",
    "STS14": "Retrieve semantically similar text",
    "STSBenchmark": "Retrieve semantically similar text",
    "FaroeseSTS": "Retrieve semantically similar text",
    "FinParaSTS": "Retrieve semantically similar text",
    "JSICK": "Retrieve semantically similar text",
    "IndicCrosslingualSTS": "Retrieve semantically similar text",
    "SemRel24STS": "Retrieve semantically similar text",
    "STS17": "Retrieve semantically similar text",
    "STS22.v2": "Retrieve semantically similar text",
    "STSES": "Retrieve semantically similar text",
    "STSB": "Retrieve semantically similar text",
    "AILAStatutes": "Given a query, retrieve documents that answer the query.",
    "HagridRetrieval": "Given a query, retrieve documents that answer the query.",
    "LegalBenchCorporateLobbying": "Given a query, retrieve documents that answer the query.",
    "LEMBPasskeyRetrieval": "Given a query, retrieve documents that answer the query.",
    "BelebeleRetrieval": "Given a query, retrieve documents that answer the query.",
    "MLQARetrieval": "Given a query, retrieve documents that answer the query.",
    "StatcanDialogueDatasetRetrieval": "Given a query, retrieve documents that answer the query.",
    "WikipediaRetrievalMultilingual": "Given a query, retrieve documents that answer the query.",
    "Core17InstructionRetrieval": "Given a query, retrieve documents that answer the query.",
    "News21InstructionRetrieval": "Given a query, retrieve documents that answer the query.",
    "Robust04InstructionRetrieval": "Given a query, retrieve documents that answer the query.",
    "WebLINXCandidatesReranking": "Given a query, retrieve documents that answer the query.",
    "WikipediaRerankingMultilingual": "Given a query, retrieve documents that answer the query.",
    "STS15": "Retrieve semantically similar text",
    "MIRACLRetrievalHardNegatives": "Given a question, retrieve passages that answer the question",
    "BIOSSES": "Retrieve semantically similar text",
    "CQADupstackRetrieval": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question",
    "CQADupstackGamingRetrieval": {
        "query": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question",
        "passage": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question",
    },
    "CQADupstackUnixRetrieval": {
        "query": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question",
        "passage": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question",
    },
    "STS16": "Retrieve semantically similar text",
    "SummEval": "Retrieve semantically similar text",
    "ATEC": "Retrieve semantically similar text",
    "BornholmBitextMining": "Retrieve parallel sentences between Danish and Bornholmsk dialect",
    "CEDRClassification": "Classify the emotion expressed in the given text into one of five categories: joy, sadness, surprise, fear, or anger",
    "DalajClassification": "Classify the linguistic acceptability of the given Swedish sentence",
    "NorwegianCourtsBitextMining": "Retrieve parallel sentences between Norwegian Bokmål and Nynorsk",
    "ScalaClassification": "Classify the linguistic acceptability of the given Scandinavian sentence",
    "SpartQA": "Given a spatial reasoning question, retrieve the passage that answers the question",
    "SwednClusteringP2P": "Identify the topic or theme of the given Swedish news articles",
    "TempReasonL1": "Given a temporal reasoning question, retrieve the passage that answers the question",
    "TwitterHjerneRetrieval": "Given a Danish question, retrieve the corresponding answer",
    "WinoGrande": "Given a commonsense reasoning question, retrieve the passage that answers the question",
    "NordicLangClassification": "Given a text, classify its Nordic language",
    "VoyageMMarcoReranking": "Given a Japanese query, retrieve relevant passages that answer the query",
    # MIEB tasks
    "AROCocoOrder": "Compositionality Evaluation of images to their captions.Each capation has four hard negatives created by order permutations.",
    "AROFlickrOrder": "Compositionality Evaluation of images to their captions.Each capation has four hard negatives created by order permutations.",
    "BLINKIT2IMultiChoice": "Retrieve images based on images and specific retrieval instructions.",
    "Country211ZeroShot": "Classifying images of 211 countries.",
    "CVBenchRelation": "decide the relation of the objects in the image.",
    "FER2013ZeroShot": "Classifying facial emotions.",
    "VidoreDocVQARetrieval": "Retrieve associated pages according to questions.",
    "VidoreShiftProjectRetrieval": "Retrieve associated pages according to questions.",
    "VidoreSyntheticDocQAAIRetrieval": "Retrieve associated pages according to questions.",
    "VidoreTabfquadRetrieval": "Retrieve associated pages according to questions.",
    "VidoreTatdqaRetrieval": "Retrieve associated pages according to questions.",
    "VQA2IT2TRetrieval": "Retrieve the correct answer for a question about an image.",
    "WebQAT2ITRetrieval": "Retrieve sources of information based on questions.",
    "WITT2IRetrieval": "Retrieve images based on multilingual descriptions.",
    "XM3600T2IRetrieval": "Retrieve images based on multilingual descriptions.",
    # MAEB tasks
    "CommonLanguageAgeDetection": "Age Classification. This is a stratified subsampled version of the original CommonLanguage dataset.",
    "CommonVoiceMini21T2ARetrieval": "Speech recordings with corresponding text transcriptions from CommonVoice dataset.",
    "FSD2019Kaggle": "Multilabel Audio Classification.",
    "JamAltArtistA2ARetrieval": "Given audio clip of a song (query), retrieve all songs from the same artist in the Jam-Alt-Lines dataset.",
    "JamAltLyricA2TRetrieval": "From audio clips of songs (query), retrieve corresponding textual lyric from the Jam-Alt-Lines dataset.",
    "SpeechCommandsZeroshotv0.02": "Sound Classification/Keyword Spotting Dataset. This is a set of one-second audio clips containing a single spoken English word or background noise. These words are from a small set of commands such as 'yes', 'no', and 'stop' spoken by various speakers. With a total of 10 labels/commands for keyword spotting and a total of 30 labels for other auxiliary tasks.",
    "VehicleSoundClustering": "Clustering vehicle sounds recorded from smartphones (0 (car class), 1 (truck, bus and van class), 2 (motorcycle class)).",
    "VoxPopuliAccentPairClassification": "Classifying same or different regional accent of English.",
}

POOLING_DIM = 2048
_SUFFIX_RE = re.compile(r"(HardNegatives|Retrieval|Summarization|\.v\d+)$")


def _flatten_prompts(prompts: dict) -> dict[str, str]:
    """Flatten prompt dict to the MTEB model_prompts format.

    ``{"Task": "instr"}``                → ``{"Task": "instr"}``
    ``{"Task": {"query":"q","passage":"p"}}`` → ``{"Task-query":"q", "Task-passage":"p"}``
    """
    result: dict[str, str] = {}
    for task_name, prompt in prompts.items():
        if isinstance(prompt, dict):
            for pt, text in prompt.items():
                result[f"{task_name}-{pt}"] = text
        else:
            result[task_name] = prompt
    return result


class BidirLMOmniEncoder(AbsEncoder):
    """MTEB-compatible multimodal encoder for BidirLM-Omni (text / image / audio).

    Instruction handling (matches training):
    - Queries / symmetric tasks:     ``f"Instruct: {instruction}\\nQuery: {text}"``
    - Documents in asymmetric tasks: no instruction prefix
    - Summarization:                 no instruction prefix
    - Images / audio:                never instructed
    """

    def __init__(
        self,
        model_name: str,
        revision: str | None = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        attn_implementation: str = "eager",
        trust_remote_code: bool = True,
        max_text_length: int = 1024,
        **kwargs: Any,
    ) -> None:
        self.model = SentenceTransformer(
            model_name,
            revision=revision,
            device=device,
            trust_remote_code=trust_remote_code,
            model_kwargs={"attn_implementation": attn_implementation},
        )
        self.model.eval()
        self.max_text_length = max_text_length

        self.task_prompts = _TASK_PROMPTS
        self.prompts_dict = _flatten_prompts(self.task_prompts)

    def _lookup_prompt(self, task_name: str):
        if task_name in self.task_prompts:
            return self.task_prompts[task_name]
        stripped = _SUFFIX_RE.sub("", task_name)
        stripped = _SUFFIX_RE.sub("", stripped)
        if stripped != task_name and stripped in self.task_prompts:
            return self.task_prompts[stripped]
        return None

    def _get_instruction(
        self,
        task_metadata: TaskMetadata,
        prompt_type: PromptType | None,
    ) -> str | None:
        task_type = task_metadata.type

        if task_type == "Summarization":
            return None

        entry = self._lookup_prompt(task_metadata.name)
        instruction = ""
        if entry is not None:
            if isinstance(entry, dict):
                key = prompt_type.value if prompt_type else "query"
                instruction = entry.get(key, entry.get("query", ""))
            else:
                instruction = entry

        # Asymmetric retrieval: documents get no instruction unless the prompt
        # dict explicitly provides a "passage" key.
        if (
            ("Retrieval" in task_type or "Reranking" in task_type)
            and prompt_type == PromptType.document
            and not (isinstance(entry, dict) and "passage" in entry)
        ):
            return None

        if not instruction and task_type in ("STS", "PairClassification"):
            instruction = "Retrieve semantically similar text"
        if not instruction and task_type == "BitextMining":
            instruction = "Retrieve parallel sentences"

        if not instruction:
            return None
        return f"Instruct: {instruction}\nQuery:"

    def encode(
        self,
        inputs: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> Array:
        """Implements AbsEncoder.encode with multimodal support (text, image, audio).

        Builds conversation messages from whichever modalities are present and
        delegates to SentenceTransformer.encode() via the native 'message' modality.
        Only text receives a task instruction.
        """
        ds_features = inputs.dataset.features

        active_cols = [c for c in ("image", "audio", "text") if c in ds_features and inputs.dataset[0].get(c) is not None]
        has_image = "image" in active_cols
        has_audio = "audio" in active_cols
        has_text  = "text"  in active_cols

        instruction = self._get_instruction(task_metadata, prompt_type)

        all_messages = []
        for batch in inputs:
            for i in range(len(batch[active_cols[0]])):
                content = []
                if has_image:
                    content.append({"type": "image", "image": batch["image"][i]})
                if has_audio:
                    content.append({"type": "audio", "audio": batch["audio"][i]})
                if has_text:
                    text = batch["text"][i]
                    content.append({
                        "type": "text",
                        "text": f"{instruction} {text}" if instruction else text,
                    })
                all_messages.append([{"role": "user", "content": content}])

        # Limit text length if no image/audio is present, otherwise use the model's max context length (32768 tokens).
        # This prevents truncating special tokens in multimodal which raised error while still enabling to limit text-only inputs.
        if has_text and not has_image and not has_audio:
            self.model.max_seq_length = self.max_text_length
        else:
            self.model.max_seq_length = 32768
        return self.model.encode(
            all_messages,
            batch_size=inputs.batch_size or 32,
            show_progress_bar=True,
            convert_to_numpy=True,
        )


bidirlm_omni_2_5b = ModelMeta(
    name="BidirLM/BidirLM-Omni-2.5B-Embedding",
    loader=BidirLMOmniEncoder,
    loader_kwargs=dict(
        trust_remote_code=True,
        max_text_length=1024,
    ),
    languages=BIDIRLM_OMNI_LANGUAGES,
    open_weights=True,
    revision="aa4d36bac807f24918c37b18d6110636c86ed673",
    release_date="2026-04-07",
    n_parameters=2_445_009_536,
    n_embedding_parameters=315_098_112,
    memory_usage_mb=4663,
    max_tokens=32768,
    embed_dim=POOLING_DIM,
    license="apache-2.0",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    modalities=["text", "image", "audio"],
    model_type=["dense"],
    reference="https://huggingface.co/BidirLM/BidirLM-Omni-2.5B-Embedding",
    public_training_code=None,
    public_training_data="https://huggingface.co/datasets/BidirLM/BidirLM-Omni-Contrastive",
    training_datasets=BIDIRLM_OMNI_TRAINING_DATASETS,
    citation=BIDIRLM_OMNI_CITATION,
)
