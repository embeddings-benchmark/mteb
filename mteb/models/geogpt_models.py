import torch
from functools import partial
from mteb.model_meta import ModelMeta, sentence_transformers_loader

PROMPTS = {
    "AILAStatutes": {
        "query": "Retrieve text based on user query.",
        "passage": ""
    },
    "AfriSentiClassification": {
        "query": "Classify user passages.",
        "passage": ""
    },
    "AlloProfClusteringS2S.v2": {
        "query": "Identify categories in user passages.",
        "passage": ""
    },
    "AlloprofReranking": {
        "query": "Retrieve text based on user query.",
        "passage": ""
    },
    "AmazonCounterfactualClassification": {
        "query": "Classify a given Amazon customer review text as either counterfactual or not-counterfactual",
        "passage": ""
    },
    "ArXivHierarchicalClusteringP2P": {
        "query": "Identify the main and secondary category of Arxiv papers based on the titles and abstracts",
        "passage": ""
    },
    "ArXivHierarchicalClusteringS2S": {
        "query": "Identify the main and secondary category of Arxiv papers based on the titles",
        "passage": ""
    },
    "ArguAna": {
        "query": "Given a claim, find documents that refute the claim",
        "passage": ""
    },
    "ArmenianParaphrasePC": {
        "query": "Retrieve text that are semantically similar to the given text.",
        "passage": ""
    },
    "AskUbuntuDupQuestions": {
        "query": "Retrieve duplicate questions from AskUbuntu forum",
        "passage": ""
    },
    "BIOSSES": {
        "query": "Retrieve semantically similar text",
        "passage": ""
    },
    "BUCC.v2": {
        "query": "Retrieve parallel sentences.",
        "passage": ""
    },
    "Banking77Classification": {
        "query": "Given a online banking query, find the corresponding intents",
        "passage": ""
    },
    "BelebeleRetrieval": {
        "query": "Retrieve text based on user query.",
        "passage": ""
    },
    "BibleNLPBitextMining": {
        "query": "Retrieve parallel sentences.",
        "passage": ""
    },
    "BigPatentClustering.v2": {
        "query": "Identify categories in user passages.",
        "passage": ""
    },
    "BiorxivClusteringP2P.v2": {
        "query": "Identify the main category of Biorxiv papers based on the titles and abstracts",
        "passage": ""
    },
    "BornholmBitextMining": {
        "query": "Retrieve parallel sentences.",
        "passage": ""
    },
    "BrazilianToxicTweetsClassification": {
        "query": "Classify user passages.",
        "passage": ""
    },
    "BulgarianStoreReviewSentimentClassfication": {
        "query": "Classify user passages.",
        "passage": ""
    },
    "CEDRClassification": {
        "query": "Given a comment as query, find expressed emotions (joy, sadness, surprise, fear, and anger)",
        "passage": ""
    },
    "CLSClusteringP2P.v2": {
        "query": "Identify the main category of scholar papers based on the titles and abstracts",
        "passage": ""
    },
    "CQADupstackGamingRetrieval": {
        "query": "Given a question paragraph at StackExchange, retrieve a question duplicated paragraph",
        "passage": ""
    },
    "CQADupstackUnixRetrieval": {
        "query": "Given a question paragraph at StackExchange, retrieve a question duplicated paragraph",
        "passage": ""
    },
    "CSFDSKMovieReviewSentimentClassification": {
        "query": "Classify user passages.",
        "passage": ""
    },
    "CTKFactsNLI": {
        "query": "Retrieve text that are semantically similar to the given text.",
        "passage": ""
    },
    "CataloniaTweetClassification": {
        "query": "Classify user passages.",
        "passage": ""
    },
    "ClimateFEVERHardNegatives": {
        "query": "Given a claim about climate change, retrieve documents that support or refute the claim",
        "passage": ""
    },
    "Core17InstructionRetrieval": {
        "query": "Retrieve text based on user query.",
        "passage": ""
    },
    "CovidRetrieval": {
        "query": "Given a question on COVID-19, retrieve news articles that answer the question",
        "passage": ""
    },
    "CyrillicTurkicLangClassification": {
        "query": "Classify user passages.",
        "passage": ""
    },
    "CzechProductReviewSentimentClassification": {
        "query": "Classify user passages.",
        "passage": ""
    },
    "DBpediaClassification": {
        "query": "Classify user passages.",
        "passage": ""
    },
    "DalajClassification": {
        "query": "Classify texts based on linguistic acceptability in Swedish",
        "passage": ""
    },
    "DiaBlaBitextMining": {
        "query": "Retrieve parallel sentences.",
        "passage": ""
    },
    "EstonianValenceClassification": {
        "query": "Classify user passages.",
        "passage": ""
    },
    "FEVERHardNegatives": {
        "query": "Given a claim, retrieve documents that support or refute the claim",
        "passage": ""
    },
    "FaroeseSTS": {
        "query": "Retrieve semantically similar text.",
        "passage": ""
    },
    "FiQA2018": {
        "query": "Given a financial question, retrieve user replies that best answer the question",
        "passage": ""
    },
    "FilipinoShopeeReviewsClassification": {
        "query": "Classify user passages.",
        "passage": ""
    },
    "FinParaSTS": {
        "query": "Retrieve semantically similar text.",
        "passage": ""
    },
    "FinancialPhrasebankClassification": {
        "query": "Classify user passages.",
        "passage": ""
    },
    "FloresBitextMining": {
        "query": "Retrieve parallel sentences.",
        "passage": ""
    },
    "GermanSTSBenchmark": {
        "query": "Retrieve semantically similar text.",
        "passage": ""
    },
    "GreekLegalCodeClassification": {
        "query": "Classify user passages.",
        "passage": ""
    },
    "GujaratiNewsClassification": {
        "query": "Classify user passages.",
        "passage": ""
    },
    "HALClusteringS2S.v2": {
        "query": "Identify categories in user passages.",
        "passage": ""
    },
    "HagridRetrieval": {
        "query": "Retrieve text based on user query.",
        "passage": ""
    },
    "HotpotQAHardNegatives": {
        "query": "Given a multi-hop question, retrieve documents that can help answer the question",
        "passage": ""
    },
    "IN22GenBitextMining": {
        "query": "Retrieve parallel sentences.",
        "passage": ""
    },
    "ImdbClassification": {
        "query": "Classify the sentiment expressed in the given movie review text from the IMDB dataset",
        "passage": ""
    },
    "IndicCrosslingualSTS": {
        "query": "Retrieve semantically similar text.",
        "passage": ""
    },
    "IndicGenBenchFloresBitextMining": {
        "query": "Retrieve parallel sentences.",
        "passage": ""
    },
    "IndicLangClassification": {
        "query": "Classify user passages.",
        "passage": ""
    },
    "IndonesianIdClickbaitClassification": {
        "query": "Classify user passages.",
        "passage": ""
    },
    "IsiZuluNewsClassification": {
        "query": "Classify user passages.",
        "passage": ""
    },
    "ItaCaseholdClassification": {
        "query": "Classify user passages.",
        "passage": ""
    },
    "JSICK": {
        "query": "Retrieve semantically similar text.",
        "passage": ""
    },
    "KorHateSpeechMLClassification": {
        "query": "Classify user passages.",
        "passage": ""
    },
    "KorSarcasmClassification": {
        "query": "Classify user passages.",
        "passage": ""
    },
    "KurdishSentimentClassification": {
        "query": "Classify user passages.",
        "passage": ""
    },
    "LEMBPasskeyRetrieval": {
        "query": "Retrieve text based on user query.",
        "passage": ""
    },
    "LegalBenchCorporateLobbying": {
        "query": "Retrieve text based on user query.",
        "passage": ""
    },
    "MIRACLRetrievalHardNegatives": {
        "query": "Retrieve text based on user query.",
        "passage": ""
    },
    "MLQARetrieval": {
        "query": "Retrieve text based on user query.",
        "passage": ""
    },
    "MTOPDomainClassification": {
        "query": "Classify the intent domain of the given utterance in task-oriented conversation",
        "passage": ""
    },
    "MacedonianTweetSentimentClassification": {
        "query": "Classify user passages.",
        "passage": ""
    },
    "MalteseNewsClassification": {
        "query": "Classify user passages.",
        "passage": ""
    },
    "MasakhaNEWSClassification": {
        "query": "Classify user passages.",
        "passage": ""
    },
    "MasakhaNEWSClusteringS2S": {
        "query": "Identify categories in user passages.",
        "passage": ""
    },
    "MassiveIntentClassification": {
        "query": "Given a user utterance as query, find the user intents",
        "passage": ""
    },
    "MassiveScenarioClassification": {
        "query": "Given a user utterance as query, find the user scenarios",
        "passage": ""
    },
    "MedrxivClusteringP2P.v2": {
        "query": "Identify the main category of Medrxiv papers based on the titles and abstracts",
        "passage": ""
    },
    "MedrxivClusteringS2S.v2": {
        "query": "Identify the main category of Medrxiv papers based on the titles",
        "passage": ""
    },
    "MindSmallReranking": {
        "query": "Retrieve relevant news articles based on user browsing history",
        "passage": ""
    },
    "MultiEURLEXMultilabelClassification": {
        "query": "Classify user passages.",
        "passage": ""
    },
    "MultiHateClassification": {
        "query": "Classify user passages.",
        "passage": ""
    },
    "NTREXBitextMining": {
        "query": "Retrieve parallel sentences.",
        "passage": ""
    },
    "NepaliNewsClassification": {
        "query": "Classify user passages.",
        "passage": ""
    },
    "News21InstructionRetrieval": {
        "query": "Retrieve text based on user query.",
        "passage": ""
    },
    "NollySentiBitextMining": {
        "query": "Retrieve parallel sentences.",
        "passage": ""
    },
    "NordicLangClassification": {
        "query": "Classify texts based on language",
        "passage": ""
    },
    "NorwegianCourtsBitextMining": {
        "query": "Retrieve parallel sentences in Norwegian Bokm\u00e5l and Nynorsk",
        "passage": ""
    },
    "NusaParagraphEmotionClassification": {
        "query": "Classify user passages.",
        "passage": ""
    },
    "NusaTranslationBitextMining": {
        "query": "Retrieve parallel sentences.",
        "passage": ""
    },
    "NusaX-senti": {
        "query": "Classify user passages.",
        "passage": ""
    },
    "NusaXBitextMining": {
        "query": "Retrieve parallel sentences.",
        "passage": ""
    },
    "OdiaNewsClassification": {
        "query": "Classify user passages.",
        "passage": ""
    },
    "OpusparcusPC": {
        "query": "Retrieve text that are semantically similar to the given text.",
        "passage": ""
    },
    "PAC": {
        "query": "Classify user passages.",
        "passage": ""
    },
    "PawsXPairClassification": {
        "query": "Retrieve text that are semantically similar to the given text.",
        "passage": ""
    },
    "PlscClusteringP2P.v2": {
        "query": "Identify categories in user passages.",
        "passage": ""
    },
    "PoemSentimentClassification": {
        "query": "Classify user passages.",
        "passage": ""
    },
    "PolEmo2.0-OUT": {
        "query": "Classify user passages.",
        "passage": ""
    },
    "PpcPC": {
        "query": "Retrieve text that are semantically similar to the given text.",
        "passage": ""
    },
    "PunjabiNewsClassification": {
        "query": "Classify user passages.",
        "passage": ""
    },
    "RTE3": {
        "query": "Retrieve text that are semantically similar to the given text.",
        "passage": ""
    },
    "Robust04InstructionRetrieval": {
        "query": "Retrieve text based on user query.",
        "passage": ""
    },
    "RomaniBibleClustering": {
        "query": "Identify categories in user passages.",
        "passage": ""
    },
    "RuBQReranking": {
        "query": "Given a question, retrieve Wikipedia passages that answer the question.",
        "passage": ""
    },
    "SCIDOCS": {
        "query": "Given a scientific paper title, retrieve paper abstracts that are cited by the given paper",
        "passage": ""
    },
    "SIB200ClusteringS2S": {
        "query": "Identify categories in user passages.",
        "passage": ""
    },
    "SICK-R": {
        "query": "Retrieve semantically similar text",
        "passage": ""
    },
    "SNLHierarchicalClusteringP2P": {
        "query": "Identify categories in a Norwegian lexicon",
        "passage": ""
    },
    "STS12": {
        "query": "Retrieve semantically similar text",
        "passage": ""
    },
    "STS13": {
        "query": "Retrieve semantically similar text",
        "passage": ""
    },
    "STS14": {
        "query": "Retrieve semantically similar text",
        "passage": ""
    },
    "STS15": {
        "query": "Retrieve semantically similar text",
        "passage": ""
    },
    "STS17": {
        "query": "Retrieve semantically similar text",
        "passage": ""
    },
    "STS22.v2": {
        "query": "Retrieve semantically similar text",
        "passage": ""
    },
    "STSB": {
        "query": "Retrieve semantically similar text.",
        "passage": ""
    },
    "STSBenchmark": {
        "query": "Retrieve semantically similar text",
        "passage": ""
    },
    "STSES": {
        "query": "Retrieve semantically similar text.",
        "passage": ""
    },
    "ScalaClassification": {
        "query": "Classify passages in Scandinavian Languages based on linguistic acceptability",
        "passage": ""
    },
    "SemRel24STS": {
        "query": "Retrieve semantically similar text.",
        "passage": ""
    },
    "SentimentAnalysisHindi": {
        "query": "Classify user passages.",
        "passage": ""
    },
    "SinhalaNewsClassification": {
        "query": "Classify user passages.",
        "passage": ""
    },
    "SiswatiNewsClassification": {
        "query": "Classify user passages.",
        "passage": ""
    },
    "SlovakMovieReviewSentimentClassification": {
        "query": "Classify user passages.",
        "passage": ""
    },
    "SpartQA": {
        "query": "Given the following spatial reasoning question, retrieve the right answer.",
        "passage": ""
    },
    "SprintDuplicateQuestions": {
        "query": "Retrieve duplicate questions from Sprint forum",
        "passage": ""
    },
    "StackExchangeClustering.v2": {
        "query": "Identify the topic or theme of StackExchange posts based on the titles",
        "passage": ""
    },
    "StackExchangeClusteringP2P.v2": {
        "query": "Identify the topic or theme of StackExchange posts based on the given paragraphs",
        "passage": ""
    },
    "StackOverflowQA": {
        "query": "Retrieve text based on user query.",
        "passage": ""
    },
    "StatcanDialogueDatasetRetrieval": {
        "query": "Retrieve text based on user query.",
        "passage": ""
    },
    "SummEvalSummarization.v2": {
        "query": "Retrieve semantically similar text",
        "passage": ""
    },
    "SwahiliNewsClassification": {
        "query": "Classify user passages.",
        "passage": ""
    },
    "SwednClusteringP2P": {
        "query": "Identify news categories in Swedish passages",
        "passage": ""
    },
    "SwissJudgementClassification": {
        "query": "Classify user passages.",
        "passage": ""
    },
    "T2Reranking": {
        "query": "Given a Chinese search query, retrieve web passages that answer the question",
        "passage": ""
    },
    "TERRa": {
        "query": "Given a premise, retrieve a hypothesis that is entailed by the premise",
        "passage": ""
    },
    "TRECCOVID": {
        "query": "Given a query, retrieve documents that answer the query",
        "passage": ""
    },
    "Tatoeba": {
        "query": "Retrieve parallel sentences.",
        "passage": ""
    },
    "TempReasonL1": {
        "query": "Given the following question about time, retrieve the correct answer.",
        "passage": ""
    },
    "Touche2020Retrieval.v3": {
        "query": "Given a question, retrieve detailed and persuasive arguments that answer the question",
        "passage": ""
    },
    "ToxicConversationsClassification": {
        "query": "Classify the given comments as either toxic or not toxic",
        "passage": ""
    },
    "TswanaNewsClassification": {
        "query": "Classify user passages.",
        "passage": ""
    },
    "TweetSentimentExtractionClassification": {
        "query": "Classify the sentiment of a given tweet as either positive, negative, or neutral",
        "passage": ""
    },
    "TweetTopicSingleClassification": {
        "query": "Classify user passages.",
        "passage": ""
    },
    "TwentyNewsgroupsClustering.v2": {
        "query": "Identify the topic or theme of the given news articles",
        "passage": ""
    },
    "TwitterHjerneRetrieval": {
        "query": "Retrieve answers to questions asked in Danish tweets",
        "passage": ""
    },
    "TwitterSemEval2015": {
        "query": "Retrieve tweets that are semantically similar to the given tweet",
        "passage": ""
    },
    "TwitterURLCorpus": {
        "query": "Retrieve tweets that are semantically similar to the given tweet",
        "passage": ""
    },
    "VoyageMMarcoReranking": {
        "query": "Given a Japanese search query, retrieve web passages that answer the question",
        "passage": ""
    },
    "WebLINXCandidatesReranking": {
        "query": "Retrieve text based on user query.",
        "passage": ""
    },
    "WikiCitiesClustering": {
        "query": "Identify categories in user passages.",
        "passage": ""
    },
    "WikiClusteringP2P.v2": {
        "query": "Identify categories in user passages.",
        "passage": ""
    },
    "WikipediaRerankingMultilingual": {
        "query": "Retrieve text based on user query.",
        "passage": ""
    },
    "WikipediaRetrievalMultilingual": {
        "query": "Retrieve text based on user query.",
        "passage": ""
    },
    "WinoGrande": {
        "query": "Given the following sentence, retrieve an appropriate answer to fill in the missing underscored part.",
        "passage": ""
    },
    "XNLI": {
        "query": "Retrieve text that are semantically similar to the given text.",
        "passage": ""
    },
    "indonli": {
        "query": "Retrieve text that are semantically similar to the given text.",
        "passage": ""
    }
}

def reformat_prompts(prompts):
    reformat = {}
    for task_name, val in prompts.items():
        for prompt_type, instruction in val.items():
            reformat[f'{task_name}-{prompt_type}'] = instruction
    return reformat


geoembedding = ModelMeta(
    name="GeoGPT-Research-Project/GeoEmbedding",
    languages=["eng-Latn"], # follows ISO 639-3 and BCP-47
    open_weights=True,
    revision="29803c28ea7ef6871194a8ebc85ad7bfe174928e",
    loader=partial(
        sentence_transformers_loader, 
        model_name_or_path="GeoGPT-Research-Project/GeoEmbedding",
        instruction_template="Instruct: {instruction}\nQuery: ",
        model_prompts=reformat_prompts(PROMPTS),
        pooling_method="lasttoken",
        mode="embedding",
        torch_dtype=torch.bfloat16,
        normalized=True,
        trust_remote_code=True
    ),
    release_date="2025-04-22",
    n_parameters=7241732096,
    memory_usage_mb=27625,
    embed_dim=4096,
    license="apache-2.0",
    max_tokens=32768,
    reference="https://huggingface.co/GeoGPT-Research-Project/GeoEmbedding",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    public_training_code="",
    public_training_data="",
    training_datasets={
        "ArguAna": ["test"], 
        "FEVER": ["train"], 
        "MSMARCO":["train"], 
        "FiQA2018": ["train"], 
        "HotpotQA": ["train"], 
        "NFCorpus": ["train"], 
        "SciFact": ["train"], 
        "AmazonCounterfactualClassification": ["train"],
        "AmazonPolarityClassification":["train"],
        "AmazonReviewsClassification": ["train"],
        "Banking77Classification": ["train"],
        "EmotionClassification": ["train"],
        "MassiveIntentClassification": ["train"],
        "MTOPDomainClassification": ["train"],
        "MTOPIntentClassification": ["train"],
        "ToxicConversationsClassification": ["train"],
        "TweetSentimentExtractionClassification": ["train"],
        "ArxivClusteringS2S": ["test"],
        "ArxivClusteringP2P": ["test"],
        "MedrixvClusteringS2S": ["test"],
        "MedrixvClusteringP2P": ["test"],
        "BiorxivClusteringS2S": ["test"],
        "BiorxivClusteringP2P": ["test"],
        "STS12": ["train"],
        "STS22": ["train"],
        "STSBenchmark": ["train"],
        "StackOverflowDupQuestions": ["train"]
    }
)

