"""Sentence models for evaluation on the Russian part of MTEB"""

from __future__ import annotations

from functools import partial

import torch

from mteb.encoder_interface import PromptType
from mteb.model_meta import ModelMeta, sentence_transformers_loader
from mteb.models.instruct_wrapper import InstructSentenceTransformerWrapper

rubert_tiny = ModelMeta(
    name="cointegrated/rubert-tiny",
    languages=["rus_Cyrl"],
    open_weights=True,
    revision="5441c5ea8026d4f6d7505ec004845409f1259fb1",
    release_date="2021-05-24",
    n_parameters=11_900_000,
    memory_usage_mb=45,
    embed_dim=312,
    license="mit",
    max_tokens=512,
    reference="https://huggingface.co/cointegrated/rubert-tiny",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    public_training_code="https://gist.github.com/avidale/7bc6350f26196918bf339c01261f5c60",
    training_datasets={
        # [Yandex Translate corpus](https://translate.yandex.ru/corpus), [OPUS-100](https://huggingface.co/datasets/opus100)
        "Tatoeba": ["train"],
    },
    adapted_from="google-bert/bert-base-multilingual-cased",
    public_training_data=None,
)

rubert_tiny2 = ModelMeta(
    name="cointegrated/rubert-tiny2",
    languages=["rus_Cyrl"],
    open_weights=True,
    revision="dad72b8f77c5eef6995dd3e4691b758ba56b90c3",
    release_date="2021-10-28",
    n_parameters=29_400_000,
    memory_usage_mb=112,
    embed_dim=312,
    license="mit",
    max_tokens=2048,
    reference="https://huggingface.co/cointegrated/rubert-tiny2",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    public_training_code="https://colab.research.google.com/drive/1mSWfIQ6PIlteLVZ9DKKpcorycgLIKZLf?usp=sharing",
    training_datasets={
        # https://huggingface.co/datasets/cointegrated/ru-paraphrase-NMT-Leipzig
        # Wikipedia https://huggingface.co/datasets/Madjogger/JamSpell_dataset
        # https://huggingface.co/datasets/imvladikon/leipzig_corpora_collection
    },
    adapted_from="cointegrated/rubert-tiny",
    public_training_data=None,
)

sbert_large_nlu_ru = ModelMeta(
    name="ai-forever/sbert_large_nlu_ru",
    languages=["rus_Cyrl"],
    open_weights=True,
    revision="af977d5dfa46a3635e29bf0ef383f2df2a08d47a",
    release_date="2020-11-20",
    n_parameters=427_000_000,
    memory_usage_mb=1629,
    embed_dim=1024,
    license="mit",
    max_tokens=512,  # best guess
    reference="https://huggingface.co/ai-forever/sbert_large_nlu_ru",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    adapted_from="google/bert_uncased_L-12_H-768_A-12",
    training_datasets={
        # SNLI
        # MNLI
    },
)

sbert_large_mt_nlu_ru = ModelMeta(
    name="ai-forever/sbert_large_mt_nlu_ru",
    languages=["rus_Cyrl"],
    open_weights=True,
    revision="05300876c2b83f46d3ddd422a7f17e45cf633bb0",
    release_date="2021-05-18",
    n_parameters=427_000_000,
    memory_usage_mb=1629,
    embed_dim=1024,
    license="not specified",
    max_tokens=512,  # best guess
    reference="https://huggingface.co/ai-forever/sbert_large_mt_nlu_ru",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets={
        # SNLI, MNLI
        # https://github.com/brmson/dataset-sts
    },
)

user_base_ru = ModelMeta(
    loader=partial(  # type: ignore
        sentence_transformers_loader,
        model_name="deepvk/USER-base",
        revision="436a489a2087d61aa670b3496a9915f84e46c861",
        model_prompts={"query": "query: ", "passage": "passage: "},
    ),
    name="deepvk/USER-base",
    languages=["rus_Cyrl"],
    open_weights=True,
    revision="436a489a2087d61aa670b3496a9915f84e46c861",
    release_date="2024-06-10",
    n_parameters=427_000_000,
    memory_usage_mb=473,
    embed_dim=768,
    license="apache-2.0",
    max_tokens=512,
    reference="https://huggingface.co/deepvk/USER-base",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    adapted_from="https://huggingface.co/deepvk/deberta-v1-base",
    use_instructions=True,
    training_datasets={
        "BibleNLPBitextMining": ["train"],
        # https://github.com/unicamp-dl/mMARCO
        # deepvk/ru-HNP
        # deepvk/ru-WANLI
        # MedNLI
        # RCB
        "TERRa": ["train"],
        # Tapaco
        # Opus100
        # BiblePar
        # RudetoxifierDataDetox
        # RuParadetox
        "MIRACL": ["train"],
        # MLDR
        # Lenta
        "MLSUMClusteringP2P": ["train"],
        "MLSUMClusteringP2P.v2": ["train"],
        "MLSUMClusteringS2S": ["train"],
        "MLSUMClusteringS2S.v2": ["train"],
        "MrTidyRetrieval": ["train"],
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
    loader=partial(  # type: ignore
        sentence_transformers_loader,
        model_name="deepvk/USER-bge-m3",
        revision="0cc6cfe48e260fb0474c753087a69369e88709ae",
    ),
    name="deepvk/USER-bge-m3",
    languages=["rus_Cyrl"],
    open_weights=True,
    revision="0cc6cfe48e260fb0474c753087a69369e88709ae",
    release_date="2024-07-05",
    n_parameters=359_026_688,
    memory_usage_mb=1370,
    embed_dim=1024,
    license="apache-2.0",
    max_tokens=8194,
    reference="https://huggingface.co/deepvk/USER-base",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    adapted_from="BAAI/bge-m3",
    use_instructions=False,
    training_datasets={
        "BibleNLPBitextMining": ["train"],
        "MLSUMClusteringP2P": ["train"],
        "MLSUMClusteringP2P.v2": ["train"],
        "MLSUMClusteringS2S": ["train"],
        "MLSUMClusteringS2S.v2": ["train"],
        # not MTEB:
        # "deepvk/ru-HNP": ["train"],
        # "deepvk/ru-WANLI": ["train"],
        # "Shitao/bge-m3-data": ["train"],
        # "RussianNLP/russian_super_glue": ["train"],
        # "reciTAL/mlsum": ["train"],
        # "Helsinki-NLP/opus-100": ["train"],
        # "Helsinki-NLP/bible_para": ["train"],
        # "d0rj/rudetoxifier_data_detox": ["train"],
        # "s-nlp/ru_paradetox": ["train"],
        # "Milana/russian_keywords": ["train"],
        # "IlyaGusev/gazeta": ["train"],
        # "d0rj/gsm8k-ru": ["train"],
        # "bragovo/dsum_ru": ["train"],
        # "CarlBrendt/Summ_Dialog_News": ["train"],
    },
    public_training_code=None,
    public_training_data=None,
)

deberta_v1_ru = ModelMeta(
    name="deepvk/deberta-v1-base",
    languages=["rus_Cyrl"],
    open_weights=True,
    revision="bdd30b0e19757e6940c92c7aff19e8fc0a60dff4",
    release_date="2023-02-07",
    n_parameters=124_000_000,
    memory_usage_mb=473,
    embed_dim=768,
    license="apache-2.0",
    max_tokens=512,
    reference="https://huggingface.co/deepvk/deberta-v1-base",
    similarity_fn_name="cosine",
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
        "WikipediaRetrievalMultilingual": [],
        "WikipediaRerankingMultilingual": [],
        "RiaNewsRetrieval": [],  # probably
    },
)

rubert_base_cased = ModelMeta(
    name="DeepPavlov/rubert-base-cased",
    languages=["rus_Cyrl"],
    open_weights=True,
    revision="4036cab694767a299f2b9e6492909664d9414229",
    release_date="2020-03-04",
    n_parameters=1280_000_000,
    memory_usage_mb=4883,
    embed_dim=768,
    license="not specified",
    max_tokens=512,
    reference="https://huggingface.co/DeepPavlov/rubert-base-cased",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    adapted_from="google/bert_uncased_L-12_H-768_A-12",
    training_datasets={
        # wikipedia
        "WikipediaRetrievalMultilingual": [],
        "WikipediaRerankingMultilingual": [],
    },
)

distilrubert_small_cased_conversational = ModelMeta(
    name="DeepPavlov/distilrubert-small-cased-conversational",
    languages=["rus_Cyrl"],
    open_weights=True,
    revision="e348066b4a7279b97138038299bddc6580a9169a",
    release_date="2022-06-28",
    n_parameters=107_000_000,
    memory_usage_mb=408,
    embed_dim=768,
    license="not specified",
    max_tokens=512,
    reference="https://huggingface.co/DeepPavlov/distilrubert-small-cased-conversational",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    adapted_from="DeepPavlov/distilrubert-base-cased-conversational",
    training_datasets={
        # OpenSubtitles[1], Dirty, Pikabu, and a Social Media segment of Taiga corpus
    },
)

rubert_base_cased_sentence = ModelMeta(
    name="DeepPavlov/rubert-base-cased-sentence",
    languages=["rus_Cyrl"],
    open_weights=True,
    revision="78b5122d6365337dd4114281b0d08cd1edbb3bc8",
    release_date="2020-03-04",
    n_parameters=107_000_000,
    memory_usage_mb=408,
    embed_dim=768,
    license="not specified",
    max_tokens=512,
    reference="https://huggingface.co/DeepPavlov/rubert-base-cased-sentence",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets={
        # "SNLI": [],
        "XNLI": ["dev"]
    },
)

labse_en_ru = ModelMeta(
    name="cointegrated/LaBSE-en-ru",
    languages=["rus_Cyrl"],
    open_weights=True,
    revision="cf0714e606d4af551e14ad69a7929cd6b0da7f7e",
    release_date="2021-06-10",
    n_parameters=129_000_000,
    memory_usage_mb=492,
    embed_dim=768,
    license="not specified",
    max_tokens=512,
    reference="https://huggingface.co/cointegrated/LaBSE-en-ru",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    public_training_code="https://colab.research.google.com/drive/1dnPRn0-ugj3vZgSpyCC9sgslM2SuSfHy?usp=sharing",
    public_training_data=None,
    training_datasets={
        # https://translate.yandex.ru/corpus
    },
    adapted_from="sentence-transformers/LaBSE",
)

turbo_models_datasets = {
    # Not MTEB: {"IlyaGusev/gazeta": ["train"], "zloelias/lenta-ru": ["train"]},
}
rubert_tiny_turbo = ModelMeta(
    name="sergeyzh/rubert-tiny-turbo",
    languages=["rus_Cyrl"],
    open_weights=True,
    revision="8ce0cf757446ce9bb2d5f5a4ac8103c7a1049054",
    release_date="2024-06-21",
    n_parameters=29_200_000,
    memory_usage_mb=111,
    embed_dim=312,
    license="mit",
    max_tokens=2048,
    reference="https://huggingface.co/sergeyzh/rubert-tiny-turbo",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=turbo_models_datasets,
    adapted_from="cointegrated/rubert-tiny2",
)

rubert_mini_frida = ModelMeta(
    name="sergeyzh/rubert-mini-frida",
    languages=["rus_Cyrl"],
    open_weights=True,
    revision="19b279b78afd945b5ccae78f63e284909814adc2",
    release_date="2025-03-02",
    n_parameters=32_300_000,
    memory_usage_mb=123,
    embed_dim=312,
    license="mit",
    max_tokens=2048,
    reference="https://huggingface.co/sergeyzh/rubert-mini-frida",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets={
        # https://huggingface.co/datasets/IlyaGusev/gazeta
        # https://huggingface.co/datasets/zloelias/lenta-ru
        # https://huggingface.co/datasets/HuggingFaceFW/fineweb-2
        # https://huggingface.co/datasets/HuggingFaceFW/fineweb
    },
    adapted_from="sergeyzh/rubert-mini-sts",
)

labse_ru_turbo = ModelMeta(
    name="sergeyzh/LaBSE-ru-turbo",
    languages=["rus_Cyrl"],
    open_weights=True,
    revision="1940b046c6b5e125df11722b899130329d0a46da",
    release_date="2024-06-27",
    n_parameters=129_000_000,
    memory_usage_mb=490,
    embed_dim=768,
    license="mit",
    max_tokens=512,
    reference="https://huggingface.co/sergeyzh/LaBSE-ru-turbo",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    training_datasets=turbo_models_datasets,
    public_training_code=None,
    adapted_from="cointegrated/LaBSE-en-ru",
    public_training_data=None,
)

berta = ModelMeta(
    name="sergeyzh/BERTA",
    languages=["rus_Cyrl"],
    open_weights=True,
    revision="914c8c8aed14042ed890fc2c662d5e9e66b2faa7",
    release_date="2025-03-10",
    n_parameters=128_000_000,
    memory_usage_mb=489,
    embed_dim=768,
    license="mit",
    max_tokens=512,
    reference="https://huggingface.co/sergeyzh/BERTA",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    training_datasets={
        # https://huggingface.co/datasets/IlyaGusev/gazeta
        # https://huggingface.co/datasets/zloelias/lenta-ru
        # https://huggingface.co/datasets/HuggingFaceFW/fineweb-2
        # https://huggingface.co/datasets/HuggingFaceFW/fineweb
    },
    public_training_code=None,
    adapted_from="sergeyzh/LaBSE-ru-turbo",
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
    f"Reranking-{PromptType.passage.value}": "search_document: ",
    "STS": "classification: ",
    "Summarization": "classification: ",
    PromptType.query.value: "search_query: ",
    PromptType.passage.value: "search_document: ",
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
    loader=partial(  # type: ignore
        sentence_transformers_loader,
        model_name="ai-forever/ru-en-RoSBERTa",
        revision="89fb1651989adbb1cfcfdedafd7d102951ad0555",
        model_prompts=rosberta_prompts,
    ),
    name="ai-forever/ru-en-RoSBERTa",
    languages=["rus_Cyrl"],
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
    similarity_fn_name="cosine",
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
        "XNLI": [],
        "XNLIV2": [],
        "LanguageClassification": [],  # XNLI
        "MIRACLReranking": ["train"],
        "MIRACLRetrieval": ["train"],
        "MIRACLRetrievalHardNegatives": ["train"],
        "MrTidyRetrieval": ["train"],
    },
    public_training_data=None,
    public_training_code=None,
    framework=["Sentence Transformers", "PyTorch"],
)

frida_prompts = {
    # Default
    "Classification": "categorize: ",
    "MultilabelClassification": "categorize: ",
    "Clustering": "categorize_topic: ",
    "PairClassification": "paraphrase: ",
    "Reranking": "paraphrase: ",
    f"Reranking-{PromptType.query.value}": "search_query: ",
    f"Reranking-{PromptType.passage.value}": "search_document: ",
    "STS": "paraphrase: ",
    "Summarization": "categorize: ",
    PromptType.query.value: "search_query: ",
    PromptType.passage.value: "search_document: ",
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
    "MIRACLReranking": ["train"],
    "MIRACLRetrieval": ["train"],
    "MIRACLRetrievalHardNegatives": ["train"],
    "MrTidyRetrieval": ["train"],
    "MSMARCO": ["train"],
    "MSMARCOHardNegatives": ["train"],
    "NanoMSMARCORetrieval": ["train"],
    "NQ": ["train"],
    "NQHardNegatives": ["train"],
    "NanoNQRetrieval": ["train"],
    # STS
    "STS12": ["train"],
    "STS22": ["train"],
    "STSBenchmark": ["train"],
    "STSBenchmarkMultilingualSTS": ["train"],  # translation not trained on
    "RUParaPhraserSTS": ["train"],
    # Classification & Clustering
    "AmazonCounterfactualClassification": ["train"],
    "AmazonPolarityClassification": ["train"],
    "AmazonReviewsClassification": ["train"],
    "ArxivClusteringP2P.v2": ["train"],
    "ArxivClusteringP2P": ["train"],
    "Banking77Classification": ["train"],
    "BiorxivClusteringP2P.v2": ["train"],
    "BiorxivClusteringP2P": ["train"],
    "CEDRClassification": ["train"],
    "DBpediaClassification": ["train"],
    "EmotionClassification": ["train"],
    "FinancialPhrasebankClassification": ["train"],
    "FrenkEnClassification": ["train"],
    "GeoreviewClassification": ["train"],
    "GeoreviewClusteringP2P": ["train"],
    "HeadlineClassification": ["train"],
    "ImdbClassification": ["train"],
    "InappropriatenessClassification": ["train"],
    "KinopoiskClassification": ["train"],
    "MasakhaNEWSClassification": ["train"],
    "MassiveIntentClassification": ["train"],
    "MassiveScenarioClassification": ["train"],
    "MedrxivClusteringP2P.v2": ["train"],
    "MedrxivClusteringP2P": ["train"],
    "MTOPDomainClassification": ["train"],
    "MTOPIntentClassification": ["train"],
    "MultiHateClassification": ["train"],
    "MultilingualSentimentClassification": ["train"],
    "NewsClassification": ["train"],
    "NusaX-senti": ["train"],
    "PoemSentimentClassification": ["train"],
    "RuReviewsClassification": ["train"],
    "RuSciBenchGRNTIClassification": ["train"],
    "RuSciBenchOECDClassification": ["train"],
    "SensitiveTopicsClassification": ["train"],
    "SIB200Classification": ["train"],
    "ToxicChatClassification": ["train"],
    "ToxicConversationsClassification": ["train"],
    "TweetSentimentClassification": ["train"],
    "TweetSentimentExtractionClassification": ["train"],
    "TweetTopicSingleClassification": ["train"],
    "YahooAnswersTopicsClassification": ["train"],
    "YelpReviewFullClassification": ["train"],
    # Pre-train sets (not mentioned above)
    # https://github.com/nomic-ai/contrastors/blob/5f7b461e5a13b5636692d1c9f1141b27232fe966/src/contrastors/configs/data/contrastive_pretrain.yaml
    # reddit_title_body
    "RedditClustering": [],
    "RedditClusteringP2P": [],
    "RedditClustering.v2": [],
    # codesearch
    "CodeSearchNetCCRetrieval": [],
    "COIRCodeSearchNetRetrieval": [],
    # stackexchange_body_body
    "StackExchangeClustering.v2": [],
    "StackExchangeClusteringP2P.v2": [],
    # wikipedia
    "WikipediaRetrievalMultilingual": [],
    "WikipediaRerankingMultilingual": [],
    # quora
    "QuoraRetrieval": [],
    "NanoQuoraRetrieval": [],
    "Quora-NL": [],  # translation not trained on
}

frida = ModelMeta(
    loader=partial(  # type: ignore
        sentence_transformers_loader,
        model_name="ai-forever/FRIDA",
        revision="7292217af9a9e6dbf07048f76b434ad1e2aa8b76",
        model_prompts=frida_prompts,
    ),
    name="ai-forever/FRIDA",
    languages=["rus_Cyrl"],
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
    similarity_fn_name="cosine",
    adapted_from="ai-forever/FRED-T5-1.7B",
    training_datasets=frida_training_datasets,
    public_training_data=None,
    public_training_code=None,
    framework=["Sentence Transformers", "PyTorch"],
)

giga_embeddings = ModelMeta(
    loader=partial(
        InstructSentenceTransformerWrapper,
        model_name="ai-sage/Giga-Embeddings-instruct",
        revision="646f5ff3587e74a18141c8d6b60d1cffd5897b92",
        trust_remote_code=True,
        instruction_template="Instruct: {instruction}\nQuery: ",
        apply_instruction_to_passages=False,
        model_kwargs={
            "torch_dtype": torch.bfloat16,
        },
    ),
    name="ai-sage/Giga-Embeddings-instruct",
    languages=["eng_Latn", "rus_Cyrl"],
    open_weights=True,
    revision="646f5ff3587e74a18141c8d6b60d1cffd5897b92",
    release_date="2024-12-13",
    n_parameters=2_530_000_000,
    memory_usage_mb=9649,
    embed_dim=2048,
    license="mit",
    max_tokens=32768,
    reference="https://huggingface.co/ai-sage/Giga-Embeddings-instruct",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
)
