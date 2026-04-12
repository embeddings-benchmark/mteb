"""MTEB model registry entry for BidirLM-Omni-2.5B-Embedding.

Copy this file to:
    mteb/models/model_implementations/bidirlm_omni_models.py

And add one import line to mteb/models/model_implementations/__init__.py:
    from .bidirlm_omni_models import bidirlm_omni_2_5b

This file is self-contained: the BidirLMOmniEncoder class and all task prompts
are embedded directly so no external JSON file is needed.
"""
from __future__ import annotations

import logging
import os
import re
import sys
from typing import TYPE_CHECKING, Any

import torch
from tqdm.auto import tqdm

from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.types import PromptType

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput

logger = logging.getLogger(__name__)

# ── Languages ─────────────────────────────────────────────────────────────────
BIDIRLM_OMNI_LANGUAGES = [
    "afr-Latn", "amh-Ethi", "ara-Arab", "aze-Latn", "bel-Cyrl", "bul-Cyrl",
    "ben-Beng", "bos-Latn", "cat-Latn", "ceb-Latn", "ces-Latn", "cym-Latn",
    "dan-Latn", "deu-Latn", "ell-Grek", "eng-Latn", "spa-Latn", "est-Latn",
    "eus-Latn", "fas-Arab", "fin-Latn", "fra-Latn", "gle-Latn", "glg-Latn",
    "guj-Gujr", "hau-Latn", "heb-Hebr", "hin-Deva", "hrv-Latn", "hat-Latn",
    "hun-Latn", "hye-Armn", "ind-Latn", "ibo-Latn", "isl-Latn", "ita-Latn",
    "jpn-Jpan", "jav-Latn", "kat-Geor", "kaz-Cyrl", "kan-Knda", "kor-Hang",
    "kir-Cyrl", "lit-Latn", "lav-Latn", "mlg-Latn", "mkd-Cyrl", "mal-Mlym",
    "mar-Deva", "msa-Latn", "mlt-Latn", "mya-Mymr", "nob-Latn", "nep-Deva",
    "nld-Latn", "nso-Latn", "nya-Latn", "pan-Guru", "pol-Latn", "pus-Arab",
    "por-Latn", "ron-Latn", "rus-Cyrl", "snd-Arab", "sin-Sinh", "slk-Latn",
    "slv-Latn", "sna-Latn", "som-Latn", "sqi-Latn", "srp-Cyrl", "sun-Latn",
    "swe-Latn", "swa-Latn", "tam-Taml", "tel-Telu", "tha-Thai", "tgl-Latn",
    "tur-Latn", "ukr-Cyrl", "urd-Arab", "vie-Latn", "wol-Latn", "xho-Latn",
    "yor-Latn", "cmn-Hans", "zul-Latn",
]

# ── Citation ──────────────────────────────────────────────────────────────────
BIDIRLM_OMNI_CITATION = """@misc{boizard2026bidirlmtextomnimodalbidirectional,
      title={BidirLM: From Text to Omnimodal Bidirectional Encoders by Adapting and Composing Causal LLMs},
      author={Nicolas Boizard and Théo Deschamps-Berger and Hippolyte Gisserot-Boukhlef and Céline Hudelot and Pierre Colombo},
      year={2026},
      eprint={2604.02045},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2604.02045},
}"""

# ── Training datasets ─────────────────────────────────────────────────────────
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

# ── Task prompts (251 entries) ────────────────────────────────────────────────
_TASK_PROMPTS: dict[str, str | dict[str, str]] = {
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
    "PAC": "Classify the sentence into one of the two types: \"BEZPIECZNE_POSTANOWIENIE_UMOWNE\" and \"KLAUZULA_ABUZYWNA\"",
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
}

# ── Encoder ───────────────────────────────────────────────────────────────────

POOLING_DIM = 2048
_SUFFIX_RE = re.compile(r"(HardNegatives|Retrieval|Summarization|\.v\d+)$")


def _flatten_prompts(prompts: dict) -> dict[str, str]:
    """Flatten nemotron prompt dict to the MTEB model_prompts format.

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
        checkpoint_path: str | None = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_sequence_length: int = 1024,
        max_image_size: int = 1024,
        attn_implementation: str = "eager",
        task_prompts: dict | None = None,
        **kwargs: Any,
    ) -> None:
        from sentence_transformers import SentenceTransformer
        from sentence_transformers import models as st_models

        from huggingface_hub import snapshot_download
        model_path_resolved = snapshot_download(repo_id=model_name, revision=revision)
        if model_path_resolved not in sys.path:
            sys.path.insert(0, model_path_resolved)
        from input_module_bidirlm_omni import BidirLMOmniInputModule

        input_module = BidirLMOmniInputModule(
            model_name=model_name,
            trust_remote_code=True,
            max_sequence_length=max_sequence_length,
            attn_implementation=attn_implementation,
        )
        pooling = st_models.Pooling(POOLING_DIM, pooling_mode="mean")
        self.model = SentenceTransformer(
            modules=[input_module, pooling], trust_remote_code=True
        )

        if checkpoint_path:
            from safetensors.torch import load_file
            weights = load_file(os.path.join(checkpoint_path, "model.safetensors"))
            input_module.model.load_state_dict(weights, strict=True)
            logger.info("Loaded fine-tuned weights from: %s", checkpoint_path)

        self.model.to(device)
        self.model.eval()
        self.device = device

        self.task_prompts = task_prompts or {}
        self.prompts_dict = _flatten_prompts(self.task_prompts)

    # ── Prompt helpers ────────────────────────────────────────────────────────

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

        if not instruction and hasattr(task_metadata, "description") and task_metadata.description:
            desc = task_metadata.description
            instruction = desc.get("description", "") if isinstance(desc, dict) else str(desc)

        if not instruction and task_type in ("STS", "PairClassification"):
            instruction = "Retrieve semantically similar text"
        if not instruction and task_type == "BitextMining":
            instruction = "Retrieve parallel sentences"

        return instruction or None

    @staticmethod
    def _format_instruction(instruction: str) -> str:
        return f"Instruct: {instruction}\nQuery:"

    @staticmethod
    def _prepend(texts: list[str], formatted_instr: str | None) -> list[str]:
        if not formatted_instr:
            return texts
        return [f"{formatted_instr} {t}" for t in texts]

    # ── Forward helpers ───────────────────────────────────────────────────────

    def _run_forward(self, features: dict) -> torch.Tensor:
        for k, v in features.items():
            if isinstance(v, torch.Tensor):
                features[k] = v.to(self.device)
        with torch.no_grad():
            features = self.model[0](features)
            features = self.model[1](features)
        return features["sentence_embedding"].cpu()

    # ── Modality-specific encoding ────────────────────────────────────────────

    def get_text_embeddings(
        self,
        loader: DataLoader,
        instruction: str | None = None,
        show_progress_bar: bool = True,
        **kwargs: Any,
    ) -> torch.Tensor:
        input_module = self.model[0]
        formatted = self._format_instruction(instruction) if instruction else None
        all_embeddings = []
        for batch in tqdm(loader, disable=not show_progress_bar, desc="Text"):
            texts = self._prepend(batch["text"], formatted)
            features = input_module.tokenize(texts)
            all_embeddings.append(self._run_forward(features))
        return torch.cat(all_embeddings, dim=0)

    def get_image_embeddings(
        self,
        loader: DataLoader,
        show_progress_bar: bool = True,
        **kwargs: Any,
    ) -> torch.Tensor:
        input_module = self.model[0]
        all_embeddings = []
        for batch in tqdm(loader, disable=not show_progress_bar, desc="Image"):
            features = input_module.tokenize(batch["image"])
            all_embeddings.append(self._run_forward(features))
        return torch.cat(all_embeddings, dim=0)

    def get_audio_embeddings(
        self,
        loader: DataLoader,
        show_progress_bar: bool = True,
        **kwargs: Any,
    ) -> torch.Tensor:
        input_module = self.model[0]
        all_embeddings = []
        for batch in tqdm(loader, disable=not show_progress_bar, desc="Audio"):
            features = input_module.tokenize(batch["audio"])
            all_embeddings.append(self._run_forward(features))
        return torch.cat(all_embeddings, dim=0)

    # ── AbsEncoder.encode ─────────────────────────────────────────────────────

    @staticmethod
    def _make_loader(inputs: DataLoader, modality: str) -> DataLoader:
        import torch.utils.data

        def _collate(batch):
            return {modality: [row[modality] for row in batch]}

        return torch.utils.data.DataLoader(
            inputs.dataset,
            batch_size=inputs.batch_size or 32,
            collate_fn=_collate,
        )

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
        features = inputs.dataset.features

        def _has_data(col: str) -> bool:
            if col not in features:
                return False
            first = inputs.dataset[0].get(col)
            return first is not None

        has_text  = _has_data("text")
        has_image = _has_data("image")
        has_audio = _has_data("audio")

        instruction = self._get_instruction(task_metadata, prompt_type)

        text_emb = image_emb = audio_emb = None
        encode_kw = {k: v for k, v in kwargs.items()
                     if k not in ("hf_split", "hf_subset")}

        if has_text:
            text_emb = self.get_text_embeddings(
                self._make_loader(inputs, "text"),
                instruction=instruction,
                **encode_kw,
            )
        if has_image:
            image_emb = self.get_image_embeddings(
                self._make_loader(inputs, "image"), **encode_kw
            )
        if has_audio:
            audio_emb = self.get_audio_embeddings(
                self._make_loader(inputs, "audio"), **encode_kw
            )

        present = [e for e in (text_emb, image_emb, audio_emb) if e is not None]

        if not present:
            raise ValueError(
                f"No supported modality in dataset features: {list(features)}"
            )
        if len(present) == 1:
            return present[0].to(torch.float32).numpy()

        # Fuse multiple modalities by element-wise addition
        fused = present[0]
        for e in present[1:]:
            if len(e) != len(fused):
                raise ValueError(
                    f"Cannot fuse modalities with different sample counts: "
                    f"{[len(p) for p in present]}"
                )
            fused = fused + e
        return fused.to(torch.float32).numpy()


# ── ModelMeta ─────────────────────────────────────────────────────────────────
bidirlm_omni_2_5b = ModelMeta(
    name="BidirLM/BidirLM-Omni-2.5B-Embedding",
    loader=BidirLMOmniEncoder,
    loader_kwargs=dict(
        task_prompts=_TASK_PROMPTS,
    ),
    languages=BIDIRLM_OMNI_LANGUAGES,
    open_weights=True,
    revision="386cfaac2d9df72d631e4739c8e6c700ae02226f",
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
