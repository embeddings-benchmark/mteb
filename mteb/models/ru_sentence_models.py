"""Sentence models for evaluation on the Russian part of MTEB"""

from __future__ import annotations

from functools import partial

from mteb.model_meta import ModelMeta, sentence_transformers_loader

from .bge_models import bge_m3_training_data

rubert_tiny = ModelMeta(
    name="cointegrated/rubert-tiny",
    languages=["rus_Cyrl"],
    open_weights=True,
    revision="5441c5ea8026d4f6d7505ec004845409f1259fb1",
    release_date="2021-05-24",
    n_parameters=29_400_000,
    embed_dim=312,
    license="mit",
    max_tokens=2048,
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
    embed_dim=1024,
    license="mit",
    max_tokens=512,  # best guess
    reference="https://huggingface.co/ai-forever/sbert_large_nlu_ru",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
)

sbert_large_mt_nlu_ru = ModelMeta(
    name="ai-forever/sbert_large_mt_nlu_ru",
    languages=["rus_Cyrl"],
    open_weights=True,
    revision="05300876c2b83f46d3ddd422a7f17e45cf633bb0",
    release_date="2021-05-18",
    n_parameters=427_000_000,
    embed_dim=1024,
    license="Not specified",
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
    embed_dim=1024,
    license="apache-2.0",
    max_tokens=8194,
    reference="https://huggingface.co/deepvk/USER-base",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    adapted_from="https://huggingface.co/BAAI/bge-m3",
    use_instructions=False,
    training_datasets={
        "BibleNLPBitextMining": ["train"],
        "MLSUMClusteringP2P": ["train"],
        "MLSUMClusteringP2P.v2": ["train"],
        "MLSUMClusteringS2S": ["train"],
        "MLSUMClusteringS2S.v2": ["train"],
        **bge_m3_training_data,
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
    training_datasets=None,
)

rubert_base_cased = ModelMeta(
    name="DeepPavlov/rubert-base-cased",
    languages=["rus_Cyrl"],
    open_weights=True,
    revision="4036cab694767a299f2b9e6492909664d9414229",
    release_date="2020-03-04",
    n_parameters=1280_000_000,
    embed_dim=768,
    license="Not specified",
    max_tokens=512,
    reference="https://huggingface.co/DeepPavlov/rubert-base-cased",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
)

distilrubert_small_cased_conversational = ModelMeta(
    name="DeepPavlov/distilrubert-small-cased-conversational",
    languages=["rus_Cyrl"],
    open_weights=True,
    revision="e348066b4a7279b97138038299bddc6580a9169a",
    release_date="2022-06-28",
    n_parameters=107_000_000,
    embed_dim=768,
    license="Not specified",
    max_tokens=512,
    reference="https://huggingface.co/DeepPavlov/distilrubert-small-cased-conversational",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
)

rubert_base_cased_sentence = ModelMeta(
    name="DeepPavlov/rubert-base-cased-sentence",
    languages=["rus_Cyrl"],
    open_weights=True,
    revision="78b5122d6365337dd4114281b0d08cd1edbb3bc8",
    release_date="2020-03-04",
    n_parameters=107_000_000,
    embed_dim=768,
    license="Not specified",
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
    embed_dim=768,
    license="Not specified",
    max_tokens=512,
    reference="https://huggingface.co/cointegrated/LaBSE-en-ru",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    public_training_code="https://colab.research.google.com/drive/1dnPRn0-ugj3vZgSpyCC9sgslM2SuSfHy?usp=sharing",
    public_training_data=None,
    training_datasets=None,
    adapted_from="sentence-transformers/LaBSE",
)

rubert_tiny_turbo = ModelMeta(
    name="sergeyzh/rubert-tiny-turbo",
    languages=["rus_Cyrl"],
    open_weights=True,
    revision="8ce0cf757446ce9bb2d5f5a4ac8103c7a1049054",
    release_date="2024-06-21",
    n_parameters=129_000_000,
    embed_dim=312,
    license="mit",
    max_tokens=512,
    reference="https://huggingface.co/sergeyzh/rubert-tiny-turbo",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,  # source model in unknown
    # Not MTEB: {"IlyaGusev/gazeta": ["train"], "zloelias/lenta-ru": ["train"]},
    adapted_from="cointegrated/rubert-tiny2",
)

labse_ru_turbo = ModelMeta(
    name="sergeyzh/LaBSE-ru-turbo",
    languages=["rus_Cyrl"],
    open_weights=True,
    revision="1940b046c6b5e125df11722b899130329d0a46da",
    release_date="2024-06-27",
    n_parameters=129_000_000,
    embed_dim=312,
    license="mit",
    max_tokens=512,
    reference="https://huggingface.co/sergeyzh/LaBSE-ru-turbo",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    training_datasets=None,
    # not MTEB: {"IlyaGusev/gazeta": ["train"], "zloelias/lenta-ru": ["train"]},
    public_training_code=None,
    adapted_from="cointegrated/LaBSE-en-ru",
    public_training_data=None,
)


rosberta_ru_en = ModelMeta(
    loader=partial(  # type: ignore
        sentence_transformers_loader,
        model_name="ai-forever/ru-en-RoSBERTa",
        revision="89fb1651989adbb1cfcfdedafd7d102951ad0555",
        model_prompts={
            "Classification": "classification: ",
            "Clustering": "clustering: ",
            "query": "search_query: ",
            "passage": "search_document: ",
        },
    ),
    name="ai-forever/ru-en-RoSBERTa",
    languages=["rus_Cyrl"],
    open_weights=True,
    revision="89fb1651989adbb1cfcfdedafd7d102951ad0555",
    release_date="2024-07-29",
    use_instructions=True,
    n_parameters=404_000_000,
    max_tokens=514,
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
    },
    public_training_data=None,
    public_training_code=None,
    framework=["Sentence Transformers", "PyTorch"],
)
