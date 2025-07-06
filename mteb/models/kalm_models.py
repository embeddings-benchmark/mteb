from __future__ import annotations

import logging
from collections.abc import Sequence
from functools import partial
from typing import Any

import numpy as np
import torch

from mteb.encoder_interface import PromptType
from mteb.model_meta import ModelMeta
from mteb.models.instruct_wrapper import InstructSentenceTransformerWrapper

logger = logging.getLogger(__name__)


class KALMWrapper(InstructSentenceTransformerWrapper):
    def encode(
        self,
        sentences: Sequence[str],
        *,
        task_name: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        if self.add_eos_token:
            sentences = [
                example + self.model.tokenizer.eos_token for example in sentences
            ]

        instruction = self.get_task_instruction(
            task_name, prompt_type, self.prompts_dict
        )
        # import there due to circular imports
        from mteb import get_task

        task = get_task(task_name)

        # to passage prompts won't be applied to passages
        if not self.apply_instruction_to_passages and prompt_type == PromptType.passage:
            instruction = None
            logger.info(
                f"No instruction used, because prompt type = {prompt_type.passage}"
            )

        if task.metadata.type in ["STS", "PairClassification", "Summarization"]:
            logger.info(
                f"No instruction used, because task type = {task.metadata.type}"
            )
            instruction = None

        if instruction:
            logger.info(f"Using instruction: '{instruction}' for task: '{task_name}'")

        embeddings = self.model.encode(
            sentences,
            prompt=instruction,
            **kwargs,
        )

        if isinstance(embeddings, torch.Tensor):
            # sometimes in kwargs can be return_tensors=True
            embeddings = embeddings.cpu().detach().float().numpy()
        return embeddings


kalm_training_data = {
    # from technical report
    # not in MTEB:
    # ExpertQA
    # MEDI2BGE
    # OpenOrca
    # PAQ
    # PubMedQA
    # SearchQA
    # arxiv_qa
    # rag-dataset-12000
    # CC-News
    # SQuAD 2.0
    # TriviaQA
    # WebGPT Comparisons
    # MultiNLI
    # NLLB
    # WikiAnswers
    # SimCSE NLI
    # SNLI
    # Aya Dataset
    # eli5
    # ----
    # in MTEB:
    "CodeFeedbackMT": ["train"],
    "CodeFeedbackST": ["train"],
    "ArxivClusteringP2P": ["train"],
    "ArxivClusteringS2S": ["train"],
    "ArxivClusteringP2P.v2": ["train"],
    "TRECCOVID": ["train"],
    "DBPedia": ["train"],
    "ESCIReranking": ["train"],
    "FEVER": ["train"],
    "FiQA2018": ["train"],
    "FEVERHardNegatives": ["train"],
    "NanoFEVERRetrieval": ["train"],
    "FEVER-NL": ["train"],  # translation not trained on
    "FiQA2018-NL": ["train"],  # translation not trained on
    "HotpotQA-PL": ["train"],  # translation not trained on
    "HotpotQA-NL": ["train"],  # translation not trained on
    "HotpotQAHardNegatives": ["train"],
    "MultiLongDocRetrieval": ["train"],
    "MSMARCO": ["train"],
    "MSMARCOHardNegatives": ["train"],
    "NanoMSMARCORetrieval": ["train"],
    "MSMARCO-PL": ["train"],  # translation not trained on
    "mMARCO-NL": ["train"],  # translation not trained on
    "MSMARCOv2": ["train"],
    "NFCorpus": ["train"],
    "SciFact": ["train"],
    "NQ": ["train"],
    "NQHardNegatives": ["train"],
    "NanoNQRetrieval": ["train"],
    "NQ-PL": ["train"],  # translation not trained on
    "NQ-NL": ["train"],  # translation not trained on
    "YahooAnswersTopicsClassification": ["train"],
    "ContractNLIConfidentialityOfAgreementLegalBenchClassification": ["train"],
    "ContractNLIExplicitIdentificationLegalBenchClassification": ["train"],
    "ContractNLIInclusionOfVerballyConveyedInformationLegalBenchClassification": [
        "train"
    ],
    "ContractNLILimitedUseLegalBenchClassification": ["train"],
    "ContractNLINoLicensingLegalBenchClassification": ["train"],
    "ContractNLINoticeOnCompelledDisclosureLegalBenchClassification": ["train"],
    "ContractNLIPermissibleAcquirementOfSimilarInformationLegalBenchClassification": [
        "train"
    ],
    "ContractNLIPermissibleCopyLegalBenchClassification": ["train"],
    "ContractNLIPermissibleDevelopmentOfSimilarInformationLegalBenchClassification": [
        "train"
    ],
    "ContractNLIPermissiblePostAgreementPossessionLegalBenchClassification": ["train"],
    "ContractNLIReturnOfConfidentialInformationLegalBenchClassification": ["train"],
    "ContractNLISharingWithEmployeesLegalBenchClassification": ["train"],
    "ContractNLISharingWithThirdPartiesLegalBenchClassification": ["train"],
    "ContractNLISurvivalOfObligationsLegalBenchClassification": ["train"],
    "QuoraRetrieval": ["train"],
    "NanoQuoraRetrieval": ["train"],
    "BiorxivClusteringP2P.v2": ["train"],
    "BiorxivClusteringS2S.v2": ["train"],
    "MedrxivClusteringP2P.v2": ["train"],
    "MedrxivClusteringS2S.v2": ["train"],
    "Banking77Classification": ["train"],
    "AmazonPolarityClassification": ["train"],
    "ImdbClassification": ["train"],
    "EmotionClassification": ["train"],
    "TweetSentimentExtractionClassification": ["train"],
    "ToxicConversationsClassification": ["train"],
    "MIRACLRetrieval": ["train"],
    "MIRACLRetrievalHardNegatives": ["train"],
    "MIRACLReranking": ["train"],
    "MrTidyRetrieval": ["train"],
    "PawsXPairClassification": ["train"],
    "AmazonReviewsClassification": ["train"],
    "AmazonCounterfactualClassification": ["train"],
    "MultilingualSentiment": ["train"],
    "MassiveIntentClassification": ["train"],
    "MassiveScenarioClassification": ["train"],
    "MTOPDomainClassification": ["train"],
    "MTOPIntentClassification": ["train"],
}

KaLM_task_prompts = {
    "AmazonCounterfactualClassification": "Given an Amazon review, judge whether it is counterfactual.",
    "AmazonPolarityClassification": "Classifying Amazon reviews into positive or negative sentiment",
    "AmazonReviewsClassification": "Classifying the given Amazon review into its appropriate rating category",
    "Banking77Classification": "Given an online banking query, find the corresponding intents",
    "EmotionClassification": "Classifying the emotion expressed in the given Twitter message into one of the six emotions: anger, fear, joy, love, sadness, and surprise",
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
    "JDReview": "Classifying sentiment of the customer review for iPhone into positive or negative",
    "OnlineShopping": "Classifying sentiment of the customer review into positive or negative",
    "Waimai": "Classify the customer review from a food takeaway platform into positive or negative",
    "MasakhaNEWSClassification": "Classifying the category of french news.",
    "CBD": "Classifying the sentiment of polish tweet reviews",
    "PolEmo2.0-IN": "Classifying the sentiment of in-domain (medicine and hotels) online reviews",
    "PolEmo2.0-OUT": "Classifying the sentiment of out-of-domain (products and school) online reviews",
    "AllegroReviews": "Classifying the sentiment of reviews from e-commerce marketplace Allegro",
    "PAC": 'Classifying the sentence into one of the two types: "BEZPIECZNE_POSTANOWIENIE_UMOWNE" and "KLAUZULA_ABUZYWNA"',
    "GeoreviewClassification": "Classifying the sentiment of Russian reviews.",
    "HeadlineClassification": "Classifying the topic of Russian headlines.",
    "InappropriatenessClassification": "Detecting inappropriate messages on sensitive topics",
    "KinopoiskClassification": "Classifying the sentiment of Kinopoisk reviews.",
    "RuReviewsClassification": "Classifying the sentiment of Russian product reviews.",
    "RuSciBenchGRNTIClassification": "Classifying the topic of Russian scientific papers.",
    "RuSciBenchOECDClassification": "Classifying the topic of Russian scientific papers.",
    "CEDRClassification": "Classification of sentences by emotions.",
    "SensitiveTopicsClassification": "Detecting inappropriate messages on sensitive topics.",
    "ArxivClusteringP2P": "Identify the main and secondary category of Arxiv papers based on the titles and abstracts",
    "ArxivClusteringS2S": "Identify the main and secondary category of Arxiv papers based on the titles",
    "BiorxivClusteringP2P": "Identify the main category of Biorxiv papers based on the titles and abstracts",
    "BiorxivClusteringS2S": "Identify the main category of Biorxiv papers based on the titles",
    "MedrxivClusteringP2P": "Identify the main category of Medrxiv papers based on the titles and abstracts",
    "MedrxivClusteringS2S": "Identify the main category of Medrxiv papers based on the titles",
    "RedditClustering": "Identify the topic or theme of Reddit posts based on the titles and posts",
    "RedditClusteringP2P": "Identify the topic or theme of Reddit posts based on the titles and posts",
    "StackExchangeClustering": "Identify the topic or theme of StackExchange posts based on the given paragraphs",
    "StackExchangeClusteringP2P": "Identify the topic or theme of StackExchange posts based on the given paragraphs",
    "TwentyNewsgroupsClustering": "Identify the topic or theme of the given news articles",
    "CLSClusteringS2S": "Identify the main category of scholar papers based on the titles",
    "CLSClusteringP2P": "Identify the main category of scholar papers based on the titles and abstracts",
    "ThuNewsClusteringS2S": "Identify the topic or theme of the given news articles based on the titles",
    "ThuNewsClusteringP2P": "Identify the topic or theme of the given news articles based on the titles and contents",
    "AlloProfClusteringP2P": "Identify the main category of Allo Prof document based on the titles and descriptions",
    "AlloProfClusteringS2S": "Identify the main category of Allo Prof document based on the titles",
    "HALClusteringS2S": "Identify the main category of academic passage based on the titles and contents",
    "MasakhaNEWSClusteringP2P": "Identify the topic or theme of the given news articles based on the titles and contents",
    "MasakhaNEWSClusteringS2S": "Identify the topic or theme of the given news articles based on the titles",
    "MLSUMClusteringP2P": "Identify the topic or theme of the given articles based on the titles and contents",
    "MLSUMClusteringS2S": "Identify the topic or theme of the given articles based on the titles",
    "EightTagsClustering": "Identify of headlines from social media posts in Polish into 8 categories: film, history, food, medicine, motorization, work, sport and technology",
    "GeoreviewClusteringP2P": "Identify the topic or theme of the Russian reviews.",
    "RuSciBenchGRNTIClusteringP2P": "Identify the topic or theme of the Russian articles.",
    "RuSciBenchOECDClusteringP2P": "Identify the topic or theme of the Russian articles.",
}


KaLM_X_task_prompts = {
    "Classification": "classify the query into different classes.",
    "MultilabelClassification": "Instruct: classify the query into different classes.",
    "Clustering": "classify the query into different classes.",
    "Reranking-query": "Given a query, retrieve documents that answer the query.",
    "Retrieval-query": "Given a query, retrieve documents that answer the query.",
    "InstructionRetrieval-query": "Given a query, retrieve documents that answer the query.",
    "AmazonCounterfactualClassification": "Classify a given Amazon customer review text as either counterfactual or not-counterfactual",
    "AmazonPolarityClassification": "Classify Amazon reviews into positive or negative sentiment",
    "AmazonReviewsClassification": "Classify the given Amazon review into its appropriate rating category",
    "Banking77Classification": "Given a online banking query, find the corresponding intents",
    "EmotionClassification": "Classify the emotion expressed in the given Twitter message into one of the six emotions: anger, fear, joy, love, sadness, and surprise",
    "ImdbClassification": "Classify the sentiment expressed in the given movie review text from the IMDB dataset",
    "MassiveIntentClassification": "Given a user utterance as query, find the user intents",
    "MassiveScenarioClassification": "Given a user utterance as query, find the user scenarios",
    "MTOPDomainClassification": "Classify the intent domain of the given utterance in task-oriented conversation",
    "MTOPIntentClassification": "Classify the intent of the given utterance in task-oriented conversation",
    "ToxicConversationsClassification": "Classify the given comments as either toxic or not toxic",
    "TweetSentimentExtractionClassification": "Classify the sentiment of a given tweet as either positive, negative, or neutral",
    "TNews": "Classify the fine-grained category of the given news title",
    "IFlyTek": "Given an App description text, find the appropriate fine-grained category",
    "MultilingualSentiment": "Classify sentiment of the customer review into positive, neutral, or negative",
    "JDReview": "Classify the customer review for iPhone on e-commerce platform into positive or negative",
    "OnlineShopping": "Classify the customer review for online shopping into positive or negative",
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
    "AskUbuntuDupQuestions-query": "Retrieve duplicate questions from AskUbuntu forum",
    "MindSmallReranking-query": "Retrieve relevant news articles based on user browsing history",
    "SciDocsRR-query": "Given a title of a scientific paper, retrieve the titles of other relevant papers",
    "StackOverflowDupQuestions-query": "Retrieve duplicate questions from StackOverflow forum",
    "T2Reranking-query": "Given a Chinese search query, retrieve web passages that answer the question",
    "MMarcoReranking-query": "Given a Chinese search query, retrieve web passages that answer the question",
    "CMedQAv1-reranking-query": "Given a Chinese community medical question, retrieve replies that best answer the question",
    "CMedQAv2-reranking-query": "Given a Chinese community medical question, retrieve replies that best answer the question",
    "ArguAna-query": "Given a claim, find documents that refute the claim",
    "ArguAna-passage": "Given a claim, find documents that refute the claim",
    "ClimateFEVER-query": "Given a claim about climate change, retrieve documents that support or refute the claim",
    "ClimateFEVERHardNegatives-query": "Given a claim about climate change, retrieve documents that support or refute the claim",
    "DBPedia-query": "Given a query, retrieve relevant entity descriptions from DBPedia",
    "FEVER-query": "Given a claim, retrieve documents that support or refute the claim",
    "FEVERHardNegatives-query": "Given a claim, retrieve documents that support or refute the claim",
    "FiQA2018-query": "Given a financial question, retrieve user replies that best answer the question",
    "HotpotQA-query": "Given a multi-hop question, retrieve documents that can help answer the question",
    "HotpotQAHardNegatives-query": "Given a multi-hop question, retrieve documents that can help answer the question",
    "MSMARCO-query": "Given a web search query, retrieve relevant passages that answer the query",
    "NFCorpus-query": "Given a question, retrieve relevant documents that best answer the question",
    "NQ-query": "Given a question, retrieve Wikipedia passages that answer the question",
    "QuoraRetrieval-query": "Given a question, retrieve questions that are semantically equivalent to the given question",
    "SCIDOCS-query": "Given a scientific paper title, retrieve paper abstracts that are cited by the given paper",
    "SciFact-query": "Given a scientific claim, retrieve documents that support or refute the claim",
    "Touche2020-query": "Given a question, retrieve detailed and persuasive arguments that answer the question",
    "Touche2020Retrieval.v3-query": "Given a question, retrieve detailed and persuasive arguments that answer the question",
    "TRECCOVID-query": "Given a query on COVID-19, retrieve documents that answer the query",
    "T2Retrieval-query": "Given a Chinese search query, retrieve web passages that answer the question",
    "MMarcoRetrieval-query": "Given a web search query, retrieve relevant passages that answer the query",
    "DuRetrieval-query": "Given a Chinese search query, retrieve web passages that answer the question",
    "CovidRetrieval-query": "Given a question on COVID-19, retrieve news articles that answer the question",
    "CmedqaRetrieval-query": "Given a Chinese community medical question, retrieve replies that best answer the question",
    "EcomRetrieval-query": "Given a user query from an e-commerce website, retrieve description sentences of relevant products",
    "MedicalRetrieval-query": "Given a medical question, retrieve user replies that best answer the question",
    "VideoRetrieval-query": "Given a video search query, retrieve the titles of relevant videos",
    "MasakhaNEWSClassification": "Classify the News in the given texts into one of the seven category: politics,sports,health,business,entertainment,technology,religion ",
    "AlloProfClusteringP2P": "Identify the main category of Allo Prof document based on the titles and descriptions",
    "AlloProfClusteringS2S": "Identify the topic of document titles from Allo Prof dataset",
    "HALClusteringS2S": "Identify the main category of academic passage based on the titles and contents",
    "MasakhaNEWSClusteringP2P": "Identify the topic or theme of the given news articles based on the titles and contents",
    "MasakhaNEWSClusteringS2S": "Identify the topic or theme of the given news articles based on the titles",
    "MLSUMClusteringP2P": "Identify the topic or theme of the given articles based on the titles and contents",
    "MLSUMClusteringS2S": "Identify the topic or theme of the given articles based on the titles",
    "SyntecReranking-query": "Given a question, retrieve passages that answer the question",
    "AlloprofReranking-query": "Given a question, retrieve passages that answer the question",
    "AlloprofRetrieval-query": "Given a question, retrieve passages that answer the question",
    "BSARDRetrieval-query": "Given a question, retrieve passages that answer the question",
    "SyntecRetrieval-query": "Given a question, retrieve passages that answer the question",
    "XPQARetrieval-query": "Given a question, retrieve passages that answer the question",
    "MintakaRetrieval-query": "Given a question, retrieve passages that answer the question",
    "CBD": "Classify the sentiment of polish tweet reviews",
    "PolEmo2.0-IN": "Classify the sentiment of in-domain (medicine and hotels) online reviews",
    "PolEmo2.0-OUT": "Classify the sentiment of out-of-domain (products and school) online reviews",
    "AllegroReviews": "Classify the sentiment of reviews from e-commerce marketplace Allegro",
    "PAC": 'Classify the sentence into one of the two types: "BEZPIECZNE_POSTANOWIENIE_UMOWNE" and "KLAUZULA_ABUZYWNA"',
    "EightTagsClustering": "Identify of headlines from social media posts in Polish  into 8 categories: film, history, food, medicine, motorization, work, sport and technology",
    "ArguAna-PL-query": "Given a claim, find documents that refute the claim",
    "DBPedia-PL-query": "Given a query, retrieve relevant entity descriptions from DBPedia",
    "FiQA-PL-query": "Given a financial question, retrieve user replies that best answer the question",
    "HotpotQA-PL-query": "Given a multi-hop question, retrieve documents that can help answer the question",
    "MSMARCO-PL-query": "Given a web search query, retrieve relevant passages that answer the query",
    "NFCorpus-PL-query": "Given a question, retrieve relevant documents that best answer the question",
    "NQ-PL-query": "Given a question, retrieve Wikipedia passages that answer the question",
    "Quora-PL-query": "Given a question, retrieve questions that are semantically equivalent to the given question",
    "SCIDOCS-PL-query": "Given a scientific paper title, retrieve paper abstracts that are cited by the given paper",
    "SciFact-PL-query": "Given a scientific claim, retrieve documents that support or refute the claim",
    "TRECCOVID-PL-query": "Given a query on COVID-19, retrieve documents that answer the query",
    "GeoreviewClassification": "Classify the organization rating based on the reviews",
    "HeadlineClassification": "Classify the topic or theme of the given news headline",
    "InappropriatenessClassification": "Classify the given message as either sensitive topic or not",
    "KinopoiskClassification": "Classify the sentiment expressed in the given movie review text",
    "RuReviewsClassification": "Classify product reviews into positive, negative or neutral sentiment",
    "RuSciBenchGRNTIClassification": "Classify the category of scientific papers based on the titles and abstracts",
    "RuSciBenchOECDClassification": "Classify the category of scientific papers based on the titles and abstracts",
    "GeoreviewClusteringP2P": "Identify the organization category based on the reviews",
    "RuSciBenchGRNTIClusteringP2P": "Identify the category of scientific papers based on the titles and abstracts",
    "RuSciBenchOECDClusteringP2P": "Identify the category of scientific papers based on the titles and abstracts",
    "RuBQReranking-query": "Given a question, retrieve Wikipedia passages that answer the question",
    "RiaNewsRetrieval-query": "Given a headline, retrieval relevant articles",
    "RuBQRetrieval-query": "Given a question, retrieve Wikipedia passages that answer the question",
    "AppsRetrieval-query": "Given a question about code problem, retrieval code that can solve user's problem",
    "COIRCodeSearchNetRetrieval-query": "Given a code snippet, retrieve the comment corresponding to that code.",
    "CodeEditSearchRetrieval-query": "Given a piece of code, retrieval code that in the ",
    "CodeFeedbackMT-query": "Given a question about coding, retrieval code or passage that can solve user's question",
    "CodeFeedbackST-query": "Given a question about coding, retrieval code or passage that can solve user's question",
    "CodeSearchNetCCRetrieval-query": "Given a code comment, retrieve the code snippet corresponding to that comment.",
    "CodeSearchNetRetrieval-query": "Given a code snippet, retrieve the comment corresponding to that code.",
    "CodeTransOceanContest-query": "Given a piece for code, retrieval semantically similar code",
    "CodeTransOceanDL-query": "Given a piece for code, retrieval semantically similar code",
    "CosQA-query": "Given a question about coding, retrieval code or passage that can solve user's question",
    "StackOverflowQA-query": "Given a question about coding, retrieval code or passage that can solve user's question",
    "SyntheticText2SQL-query": "Given a user's question, retrieve SQL queries that are appropriate responses to the question",
    "BulgarianStoreReviewSentimentClassfication": "Classify user reviews into positive or negative sentiment",
    "CzechProductReviewSentimentClassification": "Classify product reviews into positive or negative sentiment",
    "GreekLegalCodeClassification": "Given a greek legal text, classify its topic",
    "DBpediaClassification": "Given a Wikipedia articles, categorized it into classes based on its DBpedia ontology",
    "FinancialPhrasebankClassification": "Given financial news, categorized by sentiment into positive, negative, or neutral",
    "PoemSentimentClassification": "Gvien a poem, categorized by sentiment into positive, no_impact, negative or mixed",
    "TweetTopicSingleClassification": "Gvien a twitter, classify its topic",
    "EstonianValenceClassification": "Given a news article, categorized by sentiment into negatiivne, positiivne, neutraalne or vastuolulin",
    "FilipinoShopeeReviewsClassification": "Given a shop review, classify its rating on a scale from 1 to 5",
    "GujaratiNewsClassification": "Given a Gujarati news articles, classify ist topic",
    "SentimentAnalysisHindi": "Given a hindi text, categorized by sentiment into positive, negative or neutral",
    "IndonesianIdClickbaitClassification": "Given an Indonesian news headlines, classify its into clickbait or non-clickbait",
    "ItaCaseholdClassification": "Given a judgments, classify its topic",
    "KorSarcasmClassification": "Given a twitter, categorized it into sarcasm or not_sarcasm",
    "KurdishSentimentClassification": "Given a text, categorized by sentiment into positive or negative",
    "MacedonianTweetSentimentClassification": "Given a Macedonian tweet, categorized by sentiment into positive, negative, or neutral",
    "AfriSentiClassification": "Given a text, categorized by sentiment into positive, negative, or neutral",
    "CataloniaTweetClassification": "Given a tweet, categorized by sentiment into AGAINST, FAVOR or NEUTRAL",
    "CyrillicTurkicLangClassification": "Given a text, classify its language",
    "IndicLangClassification": "Given a text, classify its language",
    "MultiHateClassification": "Given a text, categorized by sentiment into hate or non-hate",
    "NusaParagraphEmotionClassification": "Given a paragraph, classify its emotion",
    "NusaX-senti": "Given a text, categorized by sentiment into positive or negative",
    "SwissJudgementClassification": "Given a news article, categorized it into approval or dismissal",
    "NepaliNewsClassification": "Given a news article, categorized it into business, entertainment or sports",
    "OdiaNewsClassification": "Given a news article, categorized it into business, entertainment or sports",
    "PunjabiNewsClassification": "Given a news article, categorized it into two-classes",
    "SinhalaNewsClassification": "Given a news article, categorized it into political, business, technology, sports and Entertainment",
    "CSFDSKMovieReviewSentimentClassification": "Given a movie review, classify its rating on a scale from 0 to 5",
    "SiswatiNewsClassification": "Given a news article, classify its topic",
    "SlovakMovieReviewSentimentClassification": "Given a movie review, categorized it into positive or negative",
    "SwahiliNewsClassification": "Given a news article, classify its domain",
    "TswanaNewsClassification": "Given a news article, classify its topic",
    "IsiZuluNewsClassification": "Given a news article, classify its topic",
    "WikiCitiesClustering": "Identify of Wikipedia articles of cities by country",
    "RomaniBibleClustering": "Identify verses from the Bible in Kalderash Romani by book.",
    "ArXivHierarchicalClusteringP2P": "Identify the main and secondary category of Arxiv papers based on the titles and abstracts",
    "ArXivHierarchicalClusteringS2S": "Identify the main and secondary category of Arxiv papers based on the titles",
    "BigPatentClustering.v2": "Identify the category of documents from the Big Patent dataset",
    "AlloProfClusteringS2S.v2": "Identify the topic of document titles from Allo Prof dataset",
    "HALClusteringS2S.v2": "Identify the topic of titles from HAL",
    "SIB200ClusteringS2S": "Identify the category of documents",
    "WikiClusteringP2P.v2": "Identify the category of wiki passages",
    "PlscClusteringP2P.v2": "Identify the category of titles+abstracts from Library of Science",
    "KorHateSpeechMLClassification": "Given a Korean online news comments, classify its fine-grained hate speech classes",
    "MalteseNewsClassification": "Given a maltese new, classify its topic",
    "MultiEURLEXMultilabelClassification": "Given a text, classify its topic",
    "BrazilianToxicTweetsClassification": "Given a tweet, classify its topic",
    "AILAStatutes-query": "Identifying the most relevant statutes for a given situation",
    "HagridRetrieval-query": "Retrieval the relevant passage for the given query",
    "LegalBenchCorporateLobbying-query": "Retrieval the relevant passage for the given query",
    "LEMBPasskeyRetrieval-query": "Retrieval the relevant passage for the given query",
    "BelebeleRetrieval-query": "Retrieval the relevant passage for the given query",
    "MLQARetrieval-query": "Retrieval the relevant passage for the given query",
    "StatcanDialogueDatasetRetrieval-query": "Retrieval the relevant passage for the given query",
    "WikipediaRetrievalMultilingual-query": "Retrieval the relevant passage for the given query",
    "Core17InstructionRetrieval-query": "Retrieval the relevant passage for the given query",
    "News21InstructionRetrieval-query": "Retrieval the relevant passage for the given query",
    "Robust04InstructionRetrieval-query": "Retrieval the relevant passage for the given query",
    "WebLINXCandidatesReranking-query": "Retrieval the relevant passage for the given query",
    "WikipediaRerankingMultilingual-query": "Retrieval the relevant passage for the given query",
    "MIRACLRetrievalHardNegatives-query": "Retrieval relevant passage for the given query",
    "CQADupstackRetrieval-query": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question",
    "CQADupstackGamingRetrieval-query": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question",
    "CQADupstackGamingRetrieval-passage": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question",
    "CQADupstackUnixRetrieval-query": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question",
    "CQADupstackUnixRetrieval-passage": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question",
}

KaLM_INSTRUCTION = "Instruct: {instruction} \n Query: "

HIT_TMG__KaLM_embedding_multilingual_mini_instruct_v1 = ModelMeta(
    loader=partial(  # type: ignore
        KALMWrapper,
        model_name="HIT-TMG/KaLM-embedding-multilingual-mini-instruct-v1",
        revision="45e42c89990c40aca042659133fc8b13c28634b5",
        instruction_template=KaLM_INSTRUCTION,
        max_seq_length=512,
        apply_instruction_to_passages=False,
        prompts_dict=KaLM_task_prompts,
    ),
    name="HIT-TMG/KaLM-embedding-multilingual-mini-instruct-v1",
    revision="45e42c89990c40aca042659133fc8b13c28634b5",
    release_date="2024-10-23",
    languages=["eng-Latn", "zho-Hans"],
    n_parameters=494032768,
    memory_usage_mb=1885,
    max_tokens=512,
    embed_dim=896,
    license="mit",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["PyTorch", "Sentence Transformers"],
    reference="https://huggingface.co/HIT-TMG/KaLM-embedding-multilingual-mini-instruct-v1",
    similarity_fn_name="cosine",
    use_instructions=True,
    training_datasets=kalm_training_data,  # Replace with actual dataset if available
    adapted_from="Qwen/Qwen2-0.5B",
    superseded_by=None,
)

HIT_TMG__KaLM_embedding_multilingual_mini_v1 = ModelMeta(
    name="HIT-TMG/KaLM-embedding-multilingual-mini-v1",
    revision="8a82a0cd2b322b91723e252486f7cce6fd8ac9d3",
    release_date="2024-08-27",
    languages=["eng-Latn", "zho-Hans"],
    n_parameters=494032768,
    memory_usage_mb=1885,
    max_tokens=512,
    embed_dim=896,
    license="mit",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["PyTorch"],
    reference="https://huggingface.co/HIT-TMG/KaLM-embedding-multilingual-mini-v1",
    similarity_fn_name="cosine",
    use_instructions=None,
    training_datasets=kalm_training_data,
    adapted_from="Qwen/Qwen2-0.5B",
    superseded_by=None,
)

HIT_TMG__KaLM_embedding_multilingual_mini_instruct_v1_5 = ModelMeta(
    loader=partial(  # type: ignore
        KALMWrapper,
        model_name="HIT-TMG/KaLM-embedding-multilingual-mini-instruct-v1.5",
        revision="fcff2f8a54e4cd96b7766fef1ee960a43d42bb3c",
        instruction_template=KaLM_INSTRUCTION,
        max_seq_length=512,
        apply_instruction_to_passages=False,
        prompts_dict=KaLM_task_prompts,
    ),
    name="HIT-TMG/KaLM-embedding-multilingual-mini-instruct-v1.5",
    revision="fcff2f8a54e4cd96b7766fef1ee960a43d42bb3c",
    release_date="2024-12-26",
    languages=["eng-Latn", "zho-Hans"],
    n_parameters=494032768,
    memory_usage_mb=1885,
    max_tokens=512,
    embed_dim=896,
    license="mit",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["PyTorch", "Sentence Transformers"],
    reference="https://huggingface.co/HIT-TMG/KaLM-embedding-multilingual-mini-instruct-v1.5",
    similarity_fn_name="cosine",
    use_instructions=True,
    training_datasets=kalm_training_data,
    adapted_from="HIT-TMG/KaLM-embedding-multilingual-mini-instruct-v1",
    superseded_by=None,
)


# KaLM_Embedding_X_0605 = ModelMeta(
#     loader=partial(
#         KALMWrapper,
#         model_name="KaLM-Team/KaLM-Embedding-X-0605",
#         revision="1",
#         instruction_template=KaLM_INSTRUCTION,
#         max_seq_length=512,
#         apply_instruction_to_passages=True,
#         prompts_dict=KaLM_X_task_prompts,
#     ),
#     name="KaLM-Team/KaLM-Embedding-X-0605",
#     revision="1",
#     languages=None,
#     open_weights=False,
#     release_date="2025-06-05",
#     n_parameters=9.24 * 1e9,
#     memory_usage_mb=35254,
#     max_tokens=8192,
#     embed_dim=3584,
#     license=None,
#     reference="https://github.com/KaLM-Team/KaLM-Embedding-X",
#     similarity_fn_name="cosine",
#     framework=["Sentence Transformers", "PyTorch"],
#     use_instructions=True,
#     public_training_code="https://github.com/HITsz-TMG/KaLM-Embedding",
#     public_training_data=None,
#     training_datasets=kalm_training_data,
# )
