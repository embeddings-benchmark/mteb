"""Sentence models for evaluation on the Russian part of MTEB"""

import torch

from mteb.models.instruct_wrapper import InstructSentenceTransformerModel
from mteb.models.model_meta import (
    ModelMeta,
    ScoringFunction,
)
from mteb.models.sentence_transformer_wrapper import sentence_transformers_loader
from mteb.types import PromptType

from .bge_models import bge_m3_training_data
from .nomic_models import (
    nomic_training_data,
)

GIGA_task_prompts = {
    "TERRa": "Given a premise, retrieve a hypothesis that is entailed by the premise",
    "RuSTSBenchmarkSTS": "Retrieve semantically similar text",
    "RUParaPhraserSTS": "Retrieve semantically similar text",
    "CEDRClassification": "Дан комментарий, определи выраженную в нем эмоцию (радость, грусть, удивление, страх, гнев или нейтрально)",
    "GeoreviewClassification": "Classify the organization rating based on the reviews",
    "GeoreviewClusteringP2P": "Классифицируй рейтинг организации на основе отзыва",
    "HeadlineClassification": "Классифицируй тему данного новостного заголовка",
    "InappropriatenessClassification": "Классифицируй данный комментарий как токсичный или не токсичный",
    "KinopoiskClassification": "Classify the sentiment expressed in the given movie review text",
    "RuReviewsClassification": "Classify product reviews into positive, negative or neutral sentiment",
    "RuSciBenchGRNTIClassification": "Classify the category of scientific papers based on the titles and abstracts",
    "RuSciBenchGRNTIClusteringP2P": "Классифицируй категорию научной статьи основываясь на аннотации",
    "RuSciBenchOECDClassification": "Classify the category of scientific papers based on the titles and abstracts",
    "RuSciBenchOECDClusteringP2P": "Классифицируй категорию научной статьи основываясь на аннотации",
    "SensitiveTopicsClassification": "Классифицируй чувствительную тему по запросу",
    "RuBQRetrieval": {
        "query": "Given a question, retrieve Wikipedia passages that answer the question",
        "document": "",
    },
    "RuBQReranking": {
        "query": "Given a question, retrieve Wikipedia passages that answer the question",
        "document": "",
    },
    "RiaNewsRetrieval": {
        "query": "Given a news title, retrieve relevant news article",
        "document": "",
    },
    "MIRACLReranking": {
        "query": "Given a question, retrieve Wikipedia passages that answer the question",
        "document": "",
    },
    "MIRACLRetrieval": {
        "query": "Given a question, retrieve Wikipedia passages that answer the question",
        "document": "",
    },
    "ArguAna": {
        "query": "Given a search query, retrieve passages that answer the question",
        "document": "Given a search query, retrieve passages that answer the question",
    },
    "CQADupstackAndroidRetrieval": {
        "query": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question",
        "document": "",
    },
    "CQADupstackEnglishRetrieval": {
        "query": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question",
        "document": "",
    },
    "CQADupstackGamingRetrieval": {
        "query": "Given a search query, retrieve passages that answer the question",
        "document": "Given a search query, retrieve passages that answer the question",
    },
    "CQADupstackGisRetrieval": {
        "query": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question",
        "document": "",
    },
    "CQADupstackMathematicaRetrieval": {
        "query": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question",
        "document": "",
    },
    "CQADupstackPhysicsRetrieval": {
        "query": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question",
        "document": "",
    },
    "CQADupstackProgrammersRetrieval": {
        "query": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question",
        "document": "",
    },
    "CQADupstackStatsRetrieval": {
        "query": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question",
        "document": "",
    },
    "CQADupstackTexRetrieval": {
        "query": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question",
        "document": "",
    },
    "CQADupstackUnixRetrieval": {
        "query": "Given a search query, retrieve passages that answer the question",
        "document": "Given a search query, retrieve passages that answer the question",
    },
    "CQADupstackWebmastersRetrieval": {
        "query": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question",
        "document": "",
    },
    "CQADupstackWordpressRetrieval": {
        "query": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question",
        "document": "",
    },
    "ClimateFEVER": {
        "query": "Given a claim about climate change, retrieve documents that support or refute the claim",
        "document": "",
    },
    "ClimateFEVERHardNegatives": {
        "query": "Given a search query, retrieve passages that answer the question",
        "document": "",
    },
    "DBPedia": {
        "query": "Given a query, retrieve relevant entity descriptions from DBPedia",
        "document": "",
    },
    "FEVER": {
        "query": "Given a claim, retrieve documents that support or refute the claim",
        "document": "",
    },
    "FEVERHardNegatives": {
        "query": "Given a search query, retrieve passages that answer the question",
        "document": "",
    },
    "FiQA2018": {
        "query": "Given a web search query, retrieve relevant passages that answer the query",
        "document": "",
    },
    "HotpotQA": {
        "query": "Given a multi-hop question, retrieve documents that can help answer the question",
        "document": "",
    },
    "HotpotQAHardNegatives": {
        "query": "Given a search query, retrieve passages that answer the question",
        "document": "",
    },
    "MSMARCO": {
        "query": "Given a web search query, retrieve relevant passages that answer the query",
        "document": "",
    },
    "NFdocument": {
        "query": "Given a question, retrieve relevant documents that best answer the question",
        "document": "",
    },
    "NQ": {
        "query": "Given a question, retrieve Wikipedia passages that answer the question",
        "document": "",
    },
    "QuoraRetrieval": {
        "query": "Given a question, retrieve questions that are semantically equivalent to the given question",
        "document": "",
    },
    "SCIDOCS": {
        "query": "Given a search query, retrieve passages that answer the question",
        "document": "",
    },
    "SciFact": {
        "query": "Given a scientific claim, retrieve documents that support or refute the claim",
        "document": "",
    },
    "TRECCOVID": {
        "query": "Given a search query, retrieve passages that answer the question",
        "document": "",
    },
    "Touche2020": {
        "query": "Given a question, retrieve detailed and persuasive arguments that answer the question",
        "document": "",
    },
    "Touche2020Retrieval.v3": {
        "query": "Given a search query, retrieve passages that answer the question",
        "document": "",
    },
    "BIOSSES": "Retrieve semantically similar text",
    "SICK-R": "Retrieve semantically similar text",
    "STS12": "Retrieve semantically similar text",
    "STS13": "Retrieve semantically similar text",
    "STS14": "Retrieve semantically similar text",
    "STS15": "Retrieve semantically similar text",
    "STS16": "Retrieve semantically similar text",
    "STS17": "Retrieve semantically similar text",
    "STS22": "Retrieve semantically similar text",
    "STS22.v2": "Retrieve semantically similar text",
    "STSBenchmark": "Retrieve semantically similar text",
    "SummEval": "Given a news summary, retrieve other semantically similar summaries",
    "SummEvalSummarization.v2": "Given a news summary, retrieve other semantically similar summaries",
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
    "ArxivClusteringP2P": "Identify the main and secondary category of Arxiv papers based on the titles and abstracts",
    "ArxivClusteringS2S": "Identify the main and secondary category of Arxiv papers based on the titles",
    "ArXivHierarchicalClusteringP2P": "Identify the main and secondary category of Arxiv papers based on the titles and abstracts",
    "ArXivHierarchicalClusteringS2S": "Identify the main and secondary category of Arxiv papers based on the titles",
    "BiorxivClusteringP2P": "Identify the main category of Biorxiv papers based on the titles and abstracts",
    "BiorxivClusteringS2S": "Identify the main category of Biorxiv papers based on the titles",
    "BiorxivClusteringP2P.v2": "Identify the main category of Biorxiv papers based on the titles and abstracts",
    "MedrxivClusteringP2P": "Identify the main category of Medrxiv papers based on the titles and abstract",
    "MedrxivClusteringS2S": "Identify the main category of Medrxiv papers based on the titles",
    "MedrxivClusteringP2P.v2": "Identify the main category of Medrxiv papers based on the titles and abstract",
    "MedrxivClusteringS2S.v2": "Identify the main category of Medrxiv papers based on the titles",
    "RedditClustering": "Identify the topic or theme of Reddit posts based on the titles",
    "RedditClusteringP2P": "Identify the topic or theme of Reddit posts based on the titles and posts",
    "StackExchangeClustering": "Identify the topic or theme of StackExchange posts based on the titles",
    "StackExchangeClusteringP2P": "Identify the topic or theme of StackExchange posts based on the given paragraphs",
    "StackExchangeClustering.v2": "Identify the topic or theme of StackExchange posts based on the titles",
    "StackExchangeClusteringP2P.v2": "Identify the topic or theme of StackExchange posts based on the given paragraphs",
    "TwentyNewsgroupsClustering": "Identify the topic or theme of the given news articles",
    "TwentyNewsgroupsClustering.v2": "Identify the topic or theme of the given news articles",
    "AskUbuntuDupQuestions": {
        "query": "Retrieve duplicate questions from AskUbuntu forum",
        "document": "Retrieve duplicate questions from AskUbuntu forum",
    },
    "MindSmallReranking": "Given a search query, retrieve passages that answer the question",
    "SciDocsRR": "Given a title of a scientific paper, retrieve the titles of other relevant papers",
    "StackOverflowDupQuestions": "Retrieve duplicate questions from StackOverflow forum",
    "SprintDuplicateQuestions": "Retrieve duplicate questions from Sprint forum",
    "TwitterSemEval2015": "Retrieve tweets that are semantically similar to the given tweet",
    "TwitterURLCorpus": "Retrieve tweets that are semantically similar to the given tweet",
}

rubert_tiny = ModelMeta(
    loader=sentence_transformers_loader,
    name="cointegrated/rubert-tiny",
    languages=["rus-Cyrl"],
    open_weights=True,
    revision="5441c5ea8026d4f6d7505ec004845409f1259fb1",
    release_date="2021-05-24",
    n_parameters=11_900_000,
    memory_usage_mb=45,
    embed_dim=312,
    license="mit",
    max_tokens=512,
    reference="https://huggingface.co/cointegrated/rubert-tiny",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    public_training_code="https://gist.github.com/avidale/7bc6350f26196918bf339c01261f5c60",
    training_datasets={
        # [Yandex Translate corpus](https://translate.yandex.ru/corpus), [OPUS-100](https://huggingface.co/datasets/opus100)
        "Tatoeba",
    },
    adapted_from="google-bert/bert-base-multilingual-cased",
    public_training_data=None,
)

rubert_tiny2 = ModelMeta(
    loader=sentence_transformers_loader,
    name="cointegrated/rubert-tiny2",
    languages=["rus-Cyrl"],
    open_weights=True,
    revision="dad72b8f77c5eef6995dd3e4691b758ba56b90c3",
    release_date="2021-10-28",
    n_parameters=29_400_000,
    memory_usage_mb=112,
    embed_dim=312,
    license="mit",
    max_tokens=2048,
    reference="https://huggingface.co/cointegrated/rubert-tiny2",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    public_training_code="https://colab.research.google.com/drive/1mSWfIQ6PIlteLVZ9DKKpcorycgLIKZLf?usp=sharing",
    training_datasets=set(
        # https://huggingface.co/datasets/cointegrated/ru-paraphrase-NMT-Leipzig
        # Wikipedia https://huggingface.co/datasets/Madjogger/JamSpell_dataset
        # https://huggingface.co/datasets/imvladikon/leipzig_corpora_collection
    ),
    adapted_from="cointegrated/rubert-tiny",
    public_training_data=None,
)

sbert_large_nlu_ru = ModelMeta(
    loader=sentence_transformers_loader,
    name="ai-forever/sbert_large_nlu_ru",
    languages=["rus-Cyrl"],
    open_weights=True,
    revision="af977d5dfa46a3635e29bf0ef383f2df2a08d47a",
    release_date="2020-11-20",
    n_parameters=427_000_000,
    memory_usage_mb=1629,
    embed_dim=1024,
    license="mit",
    max_tokens=512,  # best guess
    reference="https://huggingface.co/ai-forever/sbert_large_nlu_ru",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    adapted_from="google/bert_uncased_L-12_H-768_A-12",
    training_datasets=set(
        # SNLI
        # MNLI
    ),
)

sbert_large_mt_nlu_ru = ModelMeta(
    loader=sentence_transformers_loader,
    name="ai-forever/sbert_large_mt_nlu_ru",
    languages=["rus-Cyrl"],
    open_weights=True,
    revision="05300876c2b83f46d3ddd422a7f17e45cf633bb0",
    release_date="2021-05-18",
    n_parameters=427_000_000,
    memory_usage_mb=1629,
    embed_dim=1024,
    license="not specified",
    max_tokens=512,  # best guess
    reference="https://huggingface.co/ai-forever/sbert_large_mt_nlu_ru",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=set(
        # SNLI, MNLI
        # https://github.com/brmson/dataset-sts
    ),
)

user_base_ru = ModelMeta(
    loader=sentence_transformers_loader,
    loader_kwargs=dict(
        model_prompts={"query": "query: ", "document": "passage: "},
    ),
    name="deepvk/USER-base",
    languages=["rus-Cyrl"],
    open_weights=True,
    revision="436a489a2087d61aa670b3496a9915f84e46c861",
    release_date="2024-06-10",
    n_parameters=427_000_000,
    memory_usage_mb=473,
    embed_dim=768,
    license="apache-2.0",
    max_tokens=512,
    reference="https://huggingface.co/deepvk/USER-base",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch"],
    adapted_from="https://huggingface.co/deepvk/deberta-v1-base",
    use_instructions=True,
    citation="""@misc{deepvk2024user,
        title={USER: Universal Sentence Encoder for Russian},
        author={Malashenko, Boris and  Zemerov, Anton and Spirin, Egor},
        url={https://huggingface.co/datasets/deepvk/USER-base},
        publisher={Hugging Face}
        year={2024},
    }
    """,
    training_datasets={
        "BibleNLPBitextMining",
        # https://github.com/unicamp-dl/mMARCO
        # deepvk/ru-HNP
        # deepvk/ru-WANLI
        # MedNLI
        # RCB
        "TERRa",
        # Tapaco
        # Opus100
        # BiblePar
        # RudetoxifierDataDetox
        # RuParadetox
        "MIRACL",
        # MLDR
        # Lenta
        "MLSUMClusteringP2P",
        "MLSUMClusteringP2P.v2",
        "MLSUMClusteringS2S",
        "MLSUMClusteringS2S.v2",
        "MrTidyRetrieval",
        # "Panorama"
        # PravoIsrael
        # xlsum
        # Fialka-v1
        # RussianKeywords
        # Gazeta
        # Gsm8k-ru
        # DSumRu
        # SummDialogNews
    },
    public_training_code=None,
    public_training_data=None,
)

user_bge_m3 = ModelMeta(
    loader=sentence_transformers_loader,
    name="deepvk/USER-bge-m3",
    languages=["rus-Cyrl"],
    open_weights=True,
    revision="0cc6cfe48e260fb0474c753087a69369e88709ae",
    release_date="2024-07-05",
    n_parameters=359_026_688,
    memory_usage_mb=1370,
    embed_dim=1024,
    license="apache-2.0",
    max_tokens=8194,
    reference="https://huggingface.co/deepvk/USER-base",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch"],
    adapted_from="BAAI/bge-m3",
    use_instructions=False,
    training_datasets={
        "BibleNLPBitextMining",
        "MLSUMClusteringP2P",
        "MLSUMClusteringP2P.v2",
        "MLSUMClusteringS2S",
        "MLSUMClusteringS2S.v2",
        # not MTEB:
        # "deepvk/ru-HNP",
        # "deepvk/ru-WANLI",
        # "Shitao/bge-m3-data",
        # "RussianNLP/russian_super_glue",
        # "reciTAL/mlsum",
        # "Helsinki-NLP/opus-100",
        # "Helsinki-NLP/bible_para",
        # "d0rj/rudetoxifier_data_detox",
        # "s-nlp/ru_paradetox",
        # "Milana/russian_keywords",
        # "IlyaGusev/gazeta",
        # "d0rj/gsm8k-ru",
        # "bragovo/dsum_ru",
        # "CarlBrendt/Summ_Dialog_News",
    },
    public_training_code=None,
    public_training_data=None,
)

deberta_v1_ru = ModelMeta(
    loader=sentence_transformers_loader,
    name="deepvk/deberta-v1-base",
    languages=["rus-Cyrl"],
    open_weights=True,
    revision="bdd30b0e19757e6940c92c7aff19e8fc0a60dff4",
    release_date="2023-02-07",
    n_parameters=124_000_000,
    memory_usage_mb=473,
    embed_dim=768,
    license="apache-2.0",
    max_tokens=512,
    reference="https://huggingface.co/deepvk/deberta-v1-base",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    # Wikipedia, Books, Twitter comments, Pikabu, Proza.ru, Film subtitles, News websites, and Social corpus
    public_training_code=None,
    public_training_data=None,
    training_datasets={
        # 400 GB of filtered and deduplicated texts in total.
        # A mix of the following data: Wikipedia, Books, Twitter comments, Pikabu, Proza.ru,
        # Film subtitles, News websites, and Social corpus.
        # wikipedia
        "WikipediaRetrievalMultilingual",
        "WikipediaRerankingMultilingual",
        "RiaNewsRetrieval",  # probably
    },
)

rubert_base_cased = ModelMeta(
    loader=sentence_transformers_loader,
    name="DeepPavlov/rubert-base-cased",
    languages=["rus-Cyrl"],
    open_weights=True,
    revision="4036cab694767a299f2b9e6492909664d9414229",
    release_date="2020-03-04",
    n_parameters=1280_000_000,
    memory_usage_mb=4883,
    embed_dim=768,
    license="not specified",
    max_tokens=512,
    reference="https://huggingface.co/DeepPavlov/rubert-base-cased",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    adapted_from="google/bert_uncased_L-12_H-768_A-12",
    training_datasets={
        # wikipedia
        "WikipediaRetrievalMultilingual",
        "WikipediaRerankingMultilingual",
    },
    citation="""@misc{kuratov2019adaptationdeepbidirectionalmultilingual,
      title={Adaptation of Deep Bidirectional Multilingual Transformers for Russian Language},
      author={Yuri Kuratov and Mikhail Arkhipov},
      year={2019},
      eprint={1905.07213},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/1905.07213},
    }""",
)

distilrubert_small_cased_conversational = ModelMeta(
    loader=sentence_transformers_loader,
    name="DeepPavlov/distilrubert-small-cased-conversational",
    languages=["rus-Cyrl"],
    open_weights=True,
    revision="e348066b4a7279b97138038299bddc6580a9169a",
    release_date="2022-06-28",
    n_parameters=107_000_000,
    memory_usage_mb=408,
    embed_dim=768,
    license="not specified",
    max_tokens=512,
    reference="https://huggingface.co/DeepPavlov/distilrubert-small-cased-conversational",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    adapted_from="DeepPavlov/distilrubert-base-cased-conversational",
    training_datasets=set(
        # OpenSubtitles[1], Dirty, Pikabu, and a Social Media segment of Taiga corpus
    ),
    citation="""@misc{https://doi.org/10.48550/arxiv.2205.02340,
      doi = {10.48550/ARXIV.2205.02340},
      url = {https://arxiv.org/abs/2205.02340},
      author = {Kolesnikova, Alina and Kuratov, Yuri and Konovalov, Vasily and Burtsev, Mikhail},
      keywords = {Computation and Language (cs.CL), Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},
      title = {Knowledge Distillation of Russian Language Models with Reduction of Vocabulary},
      publisher = {arXiv},
      year = {2022},
      copyright = {arXiv.org perpetual, non-exclusive license}
    }""",
)

rubert_base_cased_sentence = ModelMeta(
    loader=sentence_transformers_loader,
    name="DeepPavlov/rubert-base-cased-sentence",
    languages=["rus-Cyrl"],
    open_weights=True,
    revision="78b5122d6365337dd4114281b0d08cd1edbb3bc8",
    release_date="2020-03-04",
    n_parameters=107_000_000,
    memory_usage_mb=408,
    embed_dim=768,
    license="not specified",
    max_tokens=512,
    reference="https://huggingface.co/DeepPavlov/rubert-base-cased-sentence",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=set(
        # "SNLI",
        "XNLI"
    ),
)

labse_en_ru = ModelMeta(
    loader=sentence_transformers_loader,
    name="cointegrated/LaBSE-en-ru",
    languages=["rus-Cyrl"],
    open_weights=True,
    revision="cf0714e606d4af551e14ad69a7929cd6b0da7f7e",
    release_date="2021-06-10",
    n_parameters=129_000_000,
    memory_usage_mb=492,
    embed_dim=768,
    license="not specified",
    max_tokens=512,
    reference="https://huggingface.co/cointegrated/LaBSE-en-ru",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    public_training_code="https://colab.research.google.com/drive/1dnPRn0-ugj3vZgSpyCC9sgslM2SuSfHy?usp=sharing",
    public_training_data=None,
    training_datasets=set(
        # https://translate.yandex.ru/corpus
    ),
    adapted_from="sentence-transformers/LaBSE",
)

turbo_models_datasets = set(
    # Not MTEB: {"IlyaGusev/gazeta", "zloelias/lenta-ru"},
)
rubert_tiny_turbo = ModelMeta(
    loader=sentence_transformers_loader,
    name="sergeyzh/rubert-tiny-turbo",
    languages=["rus-Cyrl"],
    open_weights=True,
    revision="8ce0cf757446ce9bb2d5f5a4ac8103c7a1049054",
    release_date="2024-06-21",
    n_parameters=29_200_000,
    memory_usage_mb=111,
    embed_dim=312,
    license="mit",
    max_tokens=2048,
    reference="https://huggingface.co/sergeyzh/rubert-tiny-turbo",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=turbo_models_datasets,
    adapted_from="cointegrated/rubert-tiny2",
)

rubert_mini_frida = ModelMeta(
    loader=sentence_transformers_loader,
    name="sergeyzh/rubert-mini-frida",
    languages=["rus-Cyrl"],
    open_weights=True,
    revision="19b279b78afd945b5ccae78f63e284909814adc2",
    release_date="2025-03-02",
    n_parameters=32_300_000,
    memory_usage_mb=123,
    embed_dim=312,
    license="mit",
    max_tokens=2048,
    reference="https://huggingface.co/sergeyzh/rubert-mini-frida",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=set(
        # https://huggingface.co/datasets/IlyaGusev/gazeta
        # https://huggingface.co/datasets/zloelias/lenta-ru
        # https://huggingface.co/datasets/HuggingFaceFW/fineweb-2
        # https://huggingface.co/datasets/HuggingFaceFW/fineweb
    ),
    adapted_from="sergeyzh/rubert-mini-sts",
)

labse_ru_turbo = ModelMeta(
    loader=sentence_transformers_loader,
    name="sergeyzh/LaBSE-ru-turbo",
    languages=["rus-Cyrl"],
    open_weights=True,
    revision="1940b046c6b5e125df11722b899130329d0a46da",
    release_date="2024-06-27",
    n_parameters=129_000_000,
    memory_usage_mb=490,
    embed_dim=768,
    license="mit",
    max_tokens=512,
    reference="https://huggingface.co/sergeyzh/LaBSE-ru-turbo",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    training_datasets=turbo_models_datasets,
    public_training_code=None,
    adapted_from="cointegrated/LaBSE-en-ru",
    public_training_data=None,
)

rosberta_prompts = {
    # Default
    "Classification": "classification: ",
    "MultilabelClassification": "classification: ",
    "Clustering": "clustering: ",
    "PairClassification": "classification: ",
    "Reranking": "classification: ",
    f"Reranking-{PromptType.query.value}": "search_query: ",
    f"Reranking-{PromptType.document.value}": "search_document: ",
    "STS": "classification: ",
    "Summarization": "classification: ",
    PromptType.query.value: "search_query: ",
    PromptType.document.value: "search_document: ",
    # Override some prompts for ruMTEB tasks
    "HeadlineClassification": "clustering: ",
    "InappropriatenessClassification": "clustering: ",
    "MassiveScenarioClassification": "clustering: ",
    "RuSciBenchGRNTIClassification": "clustering: ",
    "RuSciBenchOECDClassification": "clustering: ",
    "SensitiveTopicsClassification": "clustering: ",
    "STS22": "clustering: ",
}

rosberta_ru_en = ModelMeta(
    loader=sentence_transformers_loader,
    loader_kwargs=dict(
        model_prompts=rosberta_prompts,
    ),
    name="ai-forever/ru-en-RoSBERTa",
    languages=["rus-Cyrl"],
    open_weights=True,
    revision="89fb1651989adbb1cfcfdedafd7d102951ad0555",
    release_date="2024-07-29",
    use_instructions=True,
    reference="https://huggingface.co/ai-forever/ru-en-RoSBERTa",
    n_parameters=404_000_000,
    memory_usage_mb=1540,
    max_tokens=512,
    embed_dim=1024,
    license="mit",
    similarity_fn_name=ScoringFunction.COSINE,
    adapted_from="ai-forever/ruRoberta-large",
    training_datasets={
        # https://huggingface.co/ai-forever/ruRoberta-large
        # https://huggingface.co/datasets/IlyaGusev/yandex_q_full
        # https://huggingface.co/datasets/IlyaGusev/pikabu
        # https://huggingface.co/datasets/IlyaGusev/ru_stackoverflow
        # https://huggingface.co/datasets/IlyaGusev/habr
        # https://huggingface.co/datasets/its5Q/habr_qna
        # NewsCommentary
        # MultiParaCrawl
        "XNLI",
        "XNLIV2",
        "LanguageClassification",  # XNLI
        "MIRACLReranking",
        "MIRACLRetrieval",
        "MIRACLRetrievalHardNegatives",
        "MrTidyRetrieval",
    },
    public_training_data=None,
    public_training_code=None,
    framework=["Sentence Transformers", "PyTorch"],
    citation="""@misc{snegirev2024russianfocusedembeddersexplorationrumteb,
      title={The Russian-focused embedders' exploration: ruMTEB benchmark and Russian embedding model design},
      author={Artem Snegirev and Maria Tikhonova and Anna Maksimova and Alena Fenogenova and Alexander Abramov},
      year={2024},
      eprint={2408.12503},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2408.12503},
    }
    """,
)

frida_prompts = {
    # Default
    "Classification": "categorize: ",
    "MultilabelClassification": "categorize: ",
    "Clustering": "categorize_topic: ",
    "PairClassification": "paraphrase: ",
    "Reranking": "paraphrase: ",
    f"Reranking-{PromptType.query.value}": "search_query: ",
    f"Reranking-{PromptType.document.value}": "search_document: ",
    "STS": "paraphrase: ",
    "Summarization": "categorize: ",
    PromptType.query.value: "search_query: ",
    PromptType.document.value: "search_document: ",
    # Override some prompts for ruMTEB tasks
    "CEDRClassification": "categorize_sentiment: ",
    "GeoreviewClassification": "categorize_sentiment: ",
    "HeadlineClassification": "categorize_topic: ",
    "InappropriatenessClassification": "categorize_topic: ",
    "KinopoiskClassification": "categorize_sentiment: ",
    "MassiveIntentClassification": "paraphrase: ",
    "MassiveScenarioClassification": "paraphrase: ",
    "RuReviewsClassification": "categorize_sentiment: ",
    "RuSciBenchGRNTIClassification": "categorize_topic: ",
    "RuSciBenchOECDClassification": "categorize_topic: ",
    "SensitiveTopicsClassification": "categorize_topic: ",
    "TERRa": "categorize_entailment: ",
    "RiaNewsRetrieval": "categorize: ",
}

frida_training_datasets = {
    # Fine-tune sets
    # Retrieval
    "MIRACLReranking",
    "MIRACLRetrieval",
    "MIRACLRetrievalHardNegatives",
    "MrTidyRetrieval",
    "MSMARCO",
    "MSMARCOHardNegatives",
    "NanoMSMARCORetrieval",
    "NQ",
    "NQHardNegatives",
    "NanoNQRetrieval",
    # STS
    "STS12",
    "STS22",
    "STSBenchmark",
    "STSBenchmarkMultilingualSTS",  # translation not trained on
    "RUParaPhraserSTS",
    # Classification & Clustering
    "AmazonCounterfactualClassification",
    "AmazonPolarityClassification",
    "AmazonReviewsClassification",
    "ArxivClusteringP2P.v2",
    "ArxivClusteringP2P",
    "Banking77Classification",
    "BiorxivClusteringP2P.v2",
    "BiorxivClusteringP2P",
    "CEDRClassification",
    "DBpediaClassification",
    "EmotionClassification",
    "FinancialPhrasebankClassification",
    "FrenkEnClassification",
    "GeoreviewClassification",
    "GeoreviewClusteringP2P",
    "HeadlineClassification",
    "ImdbClassification",
    "InappropriatenessClassification",
    "KinopoiskClassification",
    "MasakhaNEWSClassification",
    "MassiveIntentClassification",
    "MassiveScenarioClassification",
    "MedrxivClusteringP2P.v2",
    "MedrxivClusteringP2P",
    "MTOPDomainClassification",
    "MTOPIntentClassification",
    "MultiHateClassification",
    "MultilingualSentimentClassification",
    "NewsClassification",
    "NusaX-senti",
    "PoemSentimentClassification",
    "RuReviewsClassification",
    "RuSciBenchGRNTIClassification",
    "RuSciBenchOECDClassification",
    "SensitiveTopicsClassification",
    "SIB200Classification",
    "ToxicChatClassification",
    "ToxicConversationsClassification",
    "TweetSentimentClassification",
    "TweetSentimentExtractionClassification",
    "TweetTopicSingleClassification",
    "YahooAnswersTopicsClassification",
    "YelpReviewFullClassification",
    # Pre-train sets (not mentioned above)
    # https://github.com/nomic-ai/contrastors/blob/5f7b461e5a13b5636692d1c9f1141b27232fe966/src/contrastors/configs/data/contrastive_pretrain.yaml
    # reddit_title_body
    "RedditClustering",
    "RedditClusteringP2P",
    "RedditClustering.v2",
    # codesearch
    "CodeSearchNetCCRetrieval",
    "COIRCodeSearchNetRetrieval",
    # stackexchange_body_body
    "StackExchangeClustering.v2",
    "StackExchangeClusteringP2P.v2",
    # wikipedia
    "WikipediaRetrievalMultilingual",
    "WikipediaRerankingMultilingual",
    # quora
    "QuoraRetrieval",
    "NanoQuoraRetrieval",
    "Quora-NL",  # translation not trained on
}

frida = ModelMeta(
    loader=sentence_transformers_loader,
    loader_kwargs=dict(
        model_prompts=frida_prompts,
    ),
    name="ai-forever/FRIDA",
    languages=["rus-Cyrl"],
    open_weights=True,
    revision="7292217af9a9e6dbf07048f76b434ad1e2aa8b76",
    release_date="2024-12-29",
    use_instructions=True,
    reference="https://huggingface.co/ai-forever/FRIDA",
    n_parameters=823_000_000,
    memory_usage_mb=3141,
    max_tokens=512,
    embed_dim=1536,
    license="mit",
    similarity_fn_name=ScoringFunction.COSINE,
    adapted_from="ai-forever/FRED-T5-1.7B",
    training_datasets=frida_training_datasets,
    public_training_data=None,
    public_training_code=None,
    framework=["Sentence Transformers", "PyTorch"],
)

giga_embeddings = ModelMeta(
    loader=InstructSentenceTransformerModel,
    loader_kwargs=dict(
        instruction_template="Instruct: {instruction}\nQuery: ",
        max_seq_length=4096,
        trust_remote_code=True,
        apply_instruction_to_passages=True,
        prompts_dict=GIGA_task_prompts,
        model_kwargs={
            "torch_dtype": torch.bfloat16,
        },
    ),
    name="ai-sage/Giga-Embeddings-instruct",
    languages=["eng-Latn", "rus-Cyrl"],
    open_weights=True,
    revision="0ad5b29bfecd806cecc9d66b927d828a736594dc",
    release_date="2025-09-23",
    n_parameters=3_227_176_961,
    memory_usage_mb=12865,
    embed_dim=2048,
    license="mit",
    max_tokens=4096,
    reference="https://huggingface.co/ai-sage/Giga-Embeddings-instruct",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
)

berta_training_datasets = (
    set(
        # https://huggingface.co/datasets/IlyaGusev/gazeta
        # https://huggingface.co/datasets/zloelias/lenta-ru
        # https://huggingface.co/datasets/HuggingFaceFW/fineweb-2
        # https://huggingface.co/datasets/HuggingFaceFW/fineweb
    )
    | frida_training_datasets
)  # distilled from FRIDA

berta = ModelMeta(
    loader=sentence_transformers_loader,
    name="sergeyzh/BERTA",
    languages=["rus-Cyrl"],
    open_weights=True,
    revision="914c8c8aed14042ed890fc2c662d5e9e66b2faa7",
    release_date="2025-03-10",
    n_parameters=128_000_000,
    memory_usage_mb=489,
    embed_dim=768,
    license="mit",
    max_tokens=512,
    reference="https://huggingface.co/sergeyzh/BERTA",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    training_datasets=berta_training_datasets,
    public_training_code=None,
    adapted_from="sergeyzh/LaBSE-ru-turbo",
    public_training_data=None,
)


user2_training_data = (
    set(
        # deepvk/cultura_ru_edu
        # AllNLI
        # nyuuzyou/fishkinet-posts
        # IlyaGusev/gazeta
        # its5Q/habr_qna
        # zloelias/lenta-ru
        # unicamp-dl/mmarco
        # deepvk/ru-HNP
        # deepvk/ru-WANLI
        # wikimedia/wikipedia
        # CarlBrendt/Summ_Dialog_News
        # RussianNLP/wikiomnia
        # its5Q/yandex-q
        # "mC4" ru
        # "CC-News" ru
        # MultiLongDocRetrieval
    )
    | nomic_training_data
    | bge_m3_training_data
)

user2_prompts = {
    # Override some prompts for ruMTEB tasks
    "HeadlineClassification": "search_query: ",
    "RuSciBenchGRNTIClassification": "clustering: ",
    "RuSciBenchOECDClassification": "clustering: ",
    "GeoreviewClusteringP2P": "search_query: ",
    "SensitiveTopicsClassification": "search_query: ",
    "STS22": "search_document: ",
    "InappropriatenessClassification": "classification: ",
    "CEDRClassification": "classification: ",
    # Default
    "Classification": "classification: ",
    "MultilabelClassification": "classification: ",
    "Clustering": "clustering: ",
    "PairClassification": "classification: ",
    "Reranking": "classification: ",
    f"Reranking-{PromptType.query.value}": "search_query: ",
    f"Reranking-{PromptType.document.value}": "search_document: ",
    "STS": "classification: ",
    "Summarization": "clustering: ",
    PromptType.query.value: "search_query: ",
    PromptType.document.value: "search_document: ",
}
user2_small = ModelMeta(
    loader=sentence_transformers_loader,
    loader_kwargs=dict(
        model_prompts=user2_prompts,
    ),
    name="deepvk/USER2-small",
    languages=["rus-Cyrl"],
    open_weights=True,
    revision="23f65b34cf7632032061f5cc66c14714e6d4cee4",
    release_date="2025-04-19",
    use_instructions=True,
    reference="https://huggingface.co/collections/deepvk/user2-6802650d7210f222ec60e05f",
    n_parameters=34_400_000,
    memory_usage_mb=131,
    max_tokens=8192,
    embed_dim=384,
    license="apache-2.0",
    similarity_fn_name="cosine",
    adapted_from="deepvk/RuModernBERT-small",
    training_datasets=user2_training_data,
    public_training_data=None,
    public_training_code="https://github.com/BlessedTatonka/some_code/tree/2899f27d51efdf4217fc6453799ff197e9792f1e",
    framework=["Sentence Transformers", "PyTorch"],
)

user2_base = ModelMeta(
    loader=sentence_transformers_loader,
    loader_kwargs=dict(
        model_prompts=user2_prompts,
    ),
    name="deepvk/USER2-base",
    languages=["rus-Cyrl"],
    open_weights=True,
    revision="0942cf96909b6d52e61f79a01e2d30c7be640b27",
    release_date="2025-04-19",
    use_instructions=True,
    reference="https://huggingface.co/collections/deepvk/user2-6802650d7210f222ec60e05f",
    n_parameters=149_000_000,
    memory_usage_mb=568,
    max_tokens=8192,
    embed_dim=768,
    license="apache-2.0",
    similarity_fn_name="cosine",
    adapted_from="deepvk/RuModernBERT-base",
    training_datasets=user2_training_data,
    public_training_data=None,
    public_training_code="https://github.com/BlessedTatonka/some_code/tree/2899f27d51efdf4217fc6453799ff197e9792f1e",
    framework=["Sentence Transformers", "PyTorch"],
)
