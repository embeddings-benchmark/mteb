"""This specifies the default instruction for tasks within MTEB. These are optional to use and some models might want to use their own instructions."""

import mteb

# prompt are derived from:
# scandinavian embedding benchmark: https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/blob/c8376f967d1294419be1d3eb41217d04cd3a65d3/src/seb/registered_models/e5_instruct_models.py
# e5 documentation: https://github.com/microsoft/unilm/blob/9c0f1ff7ca53431fe47d2637dfe253643d94185b/e5/utils.py#L106
DEFAULT_PROMPTS = {
    "STS": "Retrieve semantically similar text",
    "Summarization": "Given a news summary, retrieve other semantically similar summaries",
    "BitextMining": "Retrieve parallel sentences.",
    "Classification": "Classify user passages",
    "Clustering": "Identify categories in user passages",
    "Reranking": "Retrieve text based on user query.",
    "Retrieval": "Retrieve text based on user query.",
}


# This list is NOT comprehensive even for the tasks within MTEB
# TODO: We should probably move this prompt to the task object
TASKNAME2INSTRUCTIONS = {
    # BitextMining
    "BornholmBitextMining": "Retrieve parallel sentences in Danish and Bornholmsk",
    "NorwegianCourtsBitextMining ": "Retrieve parallel sentences in Norwegian Bokmål and Nynorsk",
    # Classification
    "AngryTweetsClassification": "Classify Danish tweets by sentiment. (positive, negative, neutral)",
    "DKHateClassification": "Classify Danish tweets based on offensiveness (offensive, not offensive)",
    "DanishPoliticalCommentsClassification": "Classify Danish political comments for sentiment",
    "DalajClassification": "Classify texts based on linguistic acceptability in Swedish",
    "LccSentimentClassification": "Classify texts based on sentiment",
    "NordicLangClassification": "Classify texts based on language",
    "MassiveIntentClassification": "Given a user utterance as query, find the user intents",
    "Massive Scenario": "Given a user utterance as query, find the user scenarios",
    "NoRecClassification": "Classify Norwegian reviews by sentiment",
    "SweRecClassification": "Classify Swedish reviews by sentiment",
    "Norwegian parliament": "Classify parliament speeches in Norwegian based on political affiliation",
    "ScalaClassification": "Classify passages in Scandinavian Languages based on linguistic acceptability",
    "AmazonCounterfactualClassification": "Classify a given Amazon customer review text as either counterfactual or not-counterfactual",
    "AmazonPolarityClassification": "Classify Amazon reviews into positive or negative sentiment",
    "AmazonReviewsClassification": "Classify the given Amazon review into its appropriate rating category",
    "Banking77Classification": "Given a online banking query, find the corresponding intents",
    "EmotionClassification": "Classify the emotion expressed in the given Twitter message into one of the six emotions: anger, fear, joy, love, sadness, and surprise",
    "ImdbClassification": "Classify the sentiment expressed in the given movie review text from the IMDB dataset",
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
    # Clustering
    "VGHierarchicalClusteringP2P": "Identify the categories (e.g. sports) of given articles in Norwegian",
    "VGHierarchicalClusteringS2S": "Identify the categories (e.g. sports) of given articles in Norwegian",
    "SNLHierarchicalClusteringP2P": "Identify categories in a Norwegian lexicon",
    "SNLHierarchicalClusteringS2S": "Identify categories in a Norwegian lexicon",
    "SwednClusteringP2P": "Identify news categories in Swedish passages",
    "SwednClusteringS2S": "Identify news categories in Swedish passages",
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
    # Reranking and pair classification
    "AskUbuntuDupQuestions": "Retrieve duplicate questions from AskUbuntu forum",
    "MindSmallReranking": "Retrieve relevant news articles based on user browsing history",
    "SciDocsRR": "Given a title of a scientific paper, retrieve the titles of other relevant papers",
    "StackOverflowDupQuestions": "Retrieve duplicate questions from StackOverflow forum",
    "SprintDuplicateQuestions": "Retrieve duplicate questions from Sprint forum",
    "TwitterSemEval2015": "Retrieve tweets that are semantically similar to the given tweet",
    "TwitterURLCorpus": "Retrieve tweets that are semantically similar to the given tweet",
    "T2Reranking": "Given a Chinese search query, retrieve web passages that answer the question",
    "MMarcoReranking": "Given a Chinese search query, retrieve web passages that answer the question",
    "CMedQAv1": "Given a Chinese community medical question, retrieve replies that best answer the question",
    "CMedQAv2": "Given a Chinese community medical question, retrieve replies that best answer the question",
    "Ocnli": "Retrieve semantically similar text.",
    "Cmnli": "Retrieve semantically similar text.",
    # Retrieval
    "TwitterHjerneRetrieval": "Retrieve answers to questions asked in Danish tweets",
    "SwednRetrieval": "Given a Swedish news headline retrieve summaries or news articles",
    "TV2Nordretrieval": "Given a summary of a Danish news article retrieve the corresponding news article",
    "DanFEVER": "Given a claim in Danish, retrieve documents that support the claim",
    "SNLRetrieval": "Given a lexicon headline in Norwegian, retrieve its article",
    "NorQuadRetrieval": "Given a question in Norwegian, retrieve the answer from Wikipedia articles",
    "SweFaqRetrieval": "Retrieve answers given questions in Swedish",
    "ArguAna": "Given a claim, find documents that refute the claim",
    "ClimateFEVER": "Given a claim about climate change, retrieve documents that support or refute the claim",
    "DBPedia": "Given a query, retrieve relevant entity descriptions from DBPedia",
    "FEVER": "Given a claim, retrieve documents that support or refute the claim",
    "FiQA2018": "Given a financial question, retrieve user replies that best answer the question",
    "HotpotQA": "Given a multi-hop question, retrieve documents that can help answer the question",
    "MSMARCO": "Given a web search query, retrieve relevant passages that answer the query",
    "NFCorpus": "Given a question, retrieve relevant documents that best answer the question",
    "NQ": "Given a question, retrieve Wikipedia passages that answer the question",
    "QuoraRetrieval": "Given a question, retrieve questions that are semantically equivalent to the given question",
    "SCIDOCS": "Given a scientific paper title, retrieve paper abstracts that are cited by the given paper",
    "SciFact": "Given a scientific claim, retrieve documents that support or refute the claim",
    "Touche2020": "Given a question, retrieve detailed and persuasive arguments that answer the question",
    "TRECCOVID": "Given a query on COVID-19, retrieve documents that answer the query",
    "T2Retrieval": "Given a Chinese search query, retrieve web passages that answer the question",
    "MMarcoRetrieval": "Given a web search query, retrieve relevant passages that answer the query",
    "DuRetrieval": "Given a Chinese search query, retrieve web passages that answer the question",
    "CovidRetrieval": "Given a question on COVID-19, retrieve news articles that answer the question",
    "CmedqaRetrieval": "Given a Chinese community medical question, retrieve replies that best answer the question",
    "EcomRetrieval": "Given a user query from an e-commerce website, retrieve description sentences of relevant products",
    "MedicalRetrieval": "Given a medical question, retrieve user replies that best answer the question",
    "VideoRetrieval": "Given a video search query, retrieve the titles of relevant videos",
}


def task_to_instruction(task_name: str) -> str:
    if task_name in TASKNAME2INSTRUCTIONS:
        return TASKNAME2INSTRUCTIONS[task_name]

    task = mteb.get_task(task_name)
    meta = task.metadata

    if meta.type in DEFAULT_PROMPTS:
        return DEFAULT_PROMPTS[meta.type]

    return ""
