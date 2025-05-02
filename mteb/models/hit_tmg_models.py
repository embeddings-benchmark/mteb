from __future__ import annotations

from functools import partial

from mteb.model_meta import ModelMeta
from mteb.models.instruct_wrapper import InstructSentenceTransformerWrapper

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

HIT_TMG_task_prompts = {
    "AmazonCounterfactualClassification": "Instruct: Given an Amazon review, judge whether it is counterfactual. \n Query:",
    "AmazonPolarityClassification": "Instruct: Classifying Amazon reviews into positive or negative sentiment \n Query: ",
    "AmazonReviewsClassification": "Instruct: Classifying the given Amazon review into its appropriate rating category \n Query: ",
    "Banking77Classification": "Instruct: Given an online banking query, find the corresponding intents \n Query: ",
    "EmotionClassification": "Instruct: Classifying the emotion expressed in the given Twitter message into one of the six emotions: anger, fear, joy, love, sadness, and surprise \n Query: ",
    "ImdbClassification": "Instruct: Classifying the sentiment expressed in the given movie review text from the IMDB dataset \n Query: ",
    "MassiveIntentClassification": "Instruct: Given a user utterance as query, find the user intents \n Query: ",
    "MassiveScenarioClassification": "Instruct: Given a user utterance as query, find the user scenarios \n Query: ",
    "MTOPDomainClassification": "Instruct: Classifying the intent domain of the given utterance in task-oriented conversation \n Query: ",
    "MTOPIntentClassification": "Instruct: Classifying the intent of the given utterance in task-oriented conversation \n Query: ",
    "ToxicConversationsClassification": "Instruct: Classifying the given comments as either toxic or not toxic \n Query: ",
    "TweetSentimentExtractionClassification": "Instruct: Classifying the sentiment of a given tweet as either positive, negative, or neutral \n Query: ",
    "TNews": "Instruct: Categorizing the given news title \n Query: ",
    "IFlyTek": "Instruct: Given an App description text, find the appropriate fine-grained category \n Query: ",
    "MultilingualSentiment": "Instruct: Classifying sentiment of the customer review into positive, neutral, or negative \n Query: ",
    "JDReview": "Instruct: Classifying sentiment of the customer review for iPhone into positive or negative \n Query: ",
    "OnlineShopping": "Instruct: Classifying sentiment of the customer review into positive or negative \n Query: ",
    "Waimai": "Instruct: Classify the customer review from a food takeaway platform into positive or negative \n Query: ",
    "MasakhaNEWSClassification": "Instruct: Classifying the category of french news. \n Query: ",
    "CBD": "Instruct: Classifying the sentiment of polish tweet reviews \n Query: ",
    "PolEmo2.0-IN": "Instruct: Classifying the sentiment of in-domain (medicine and hotels) online reviews \n Query: ",
    "PolEmo2.0-OUT": "Instruct: Classifying the sentiment of out-of-domain (products and school) online reviews \n Query: ",
    "AllegroReviews": "Instruct: Classifying the sentiment of reviews from e-commerce marketplace Allegro \n Query: ",
    "PAC": 'Instruct: Classifying the sentence into one of the two types: "BEZPIECZNE_POSTANOWIENIE_UMOWNE" and "KLAUZULA_ABUZYWNA" \n Query: ',
    "GeoreviewClassification": "Instruct: Classifying the sentiment of Russian reviews. \n Query: ",
    "HeadlineClassification": "Instruct: Classifying the topic of Russian headlines. \n Query: ",
    "InappropriatenessClassification": "Instruct: Detecting inappropriate messages on sensitive topics \n Query: ",
    "KinopoiskClassification": "Instruct: Classifying the sentiment of Kinopoisk reviews. \n Query: ",
    "RuReviewsClassification": "Instruct: Classifying the sentiment of Russian product reviews. \n Query: ",
    "RuSciBenchGRNTIClassification": "Instruct: Classifying the topic of Russian scientific papers. \n Query: ",
    "RuSciBenchOECDClassification": "Instruct: Classifying the topic of Russian scientific papers. \n Query: ",
    "CEDRClassification": "Instruct: Classification of sentences by emotions. \n Query: ",
    "SensitiveTopicsClassification": "Instruct: Detecting inappropriate messages on sensitive topics. \n Query: ",
    "ArxivClusteringP2P": "Instruct: Identify the main and secondary category of Arxiv papers based on the titles and abstracts \n Query: ",
    "ArxivClusteringS2S": "Instruct: Identify the main and secondary category of Arxiv papers based on the titles \n Query: ",
    "BiorxivClusteringP2P": "Instruct: Identify the main category of Biorxiv papers based on the titles and abstracts \n Query: ",
    "BiorxivClusteringS2S": "Instruct: Identify the main category of Biorxiv papers based on the titles \n Query: ",
    "MedrxivClusteringP2P": "Instruct: Identify the main category of Medrxiv papers based on the titles and abstracts \n Query: ",
    "MedrxivClusteringS2S": "Instruct: Identify the main category of Medrxiv papers based on the titles \n Query: ",
    "RedditClustering": "Instruct: Identify the topic or theme of Reddit posts based on the titles and posts \n Query: ",
    "RedditClusteringP2P": "Instruct: Identify the topic or theme of Reddit posts based on the titles and posts \n Query: ",
    "StackExchangeClustering": "Instruct: Identify the topic or theme of StackExchange posts based on the given paragraphs \n Query: ",
    "StackExchangeClusteringP2P": "Instruct: Identify the topic or theme of StackExchange posts based on the given paragraphs \n Query: ",
    "TwentyNewsgroupsClustering": "Instruct: Identify the topic or theme of the given news articles \n Query: ",
    "CLSClusteringS2S": "Instruct: Identify the main category of scholar papers based on the titles \n Query: ",
    "CLSClusteringP2P": "Instruct: Identify the main category of scholar papers based on the titles and abstracts \n Query: ",
    "ThuNewsClusteringS2S": "Instruct: Identify the topic or theme of the given news articles based on the titles \n Query: ",
    "ThuNewsClusteringP2P": "Instruct: Identify the topic or theme of the given news articles based on the titles and contents \n Query: ",
    "AlloProfClusteringP2P": "Instruct: Identify the main category of Allo Prof document based on the titles and descriptions \n Query: ",
    "AlloProfClusteringS2S": "Instruct: Identify the main category of Allo Prof document based on the titles \n Query: ",
    "HALClusteringS2S": "Instruct: Identify the main category of academic passage based on the titles and contents \n Query: ",
    "MasakhaNEWSClusteringP2P": "Instruct: Identify the topic or theme of the given news articles based on the titles and contents \n Query: ",
    "MasakhaNEWSClusteringS2S": "Instruct: Identify the topic or theme of the given news articles based on the titles \n Query: ",
    "MLSUMClusteringP2P": "Instruct: Identify the topic or theme of the given articles based on the titles and contents \n Query: ",
    "MLSUMClusteringS2S": "Instruct: Identify the topic or theme of the given articles based on the titles \n Query: ",
    "EightTagsClustering": "Instruct: Identify of headlines from social media posts in Polish into 8 categories: film, history, food, medicine, motorization, work, sport and technology \n Query: ",
    "GeoreviewClusteringP2P": "Instruct: Identify the topic or theme of the Russian reviews. \n Query: ",
    "RuSciBenchGRNTIClusteringP2P": "Instruct: Identify the topic or theme of the Russian articles. \n Query: ",
    "RuSciBenchOECDClusteringP2P": "Instruct: Identify the topic or theme of the Russian articles. \n Query: ",
}

HIT_TMG_INSTRUCTION = "Instruct: {instruction}\nQuery: "

HIT_TMG__KaLM_embedding_multilingual_mini_instruct_v1 = ModelMeta(
    loader=partial(  # type: ignore
        InstructSentenceTransformerWrapper,
        model_name="HIT-TMG/KaLM-embedding-multilingual-mini-instruct-v1",
        revision="45e42c89990c40aca042659133fc8b13c28634b5",
        instruction_template=HIT_TMG_INSTRUCTION,
        max_seq_length=512,
        apply_instruction_to_passages=True,
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
        InstructSentenceTransformerWrapper,
        model_name="HIT-TMG/KaLM-embedding-multilingual-mini-instruct-v1.5",
        revision="fcff2f8a54e4cd96b7766fef1ee960a43d42bb3c",
        instruction_template=HIT_TMG_INSTRUCTION,
        max_seq_length=512,
        apply_instruction_to_passages=True,
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
