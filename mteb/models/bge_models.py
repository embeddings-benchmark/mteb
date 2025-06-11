from __future__ import annotations

from functools import partial

from mteb.model_meta import ModelMeta, sentence_transformers_loader
from mteb.models.e5_instruct import E5_MISTRAL_TRAINING_DATA

model_prompts = {"query": "Represent this sentence for searching relevant passages: "}
model_prompts_zh = {"query": "为这个句子生成表示以用于检索相关文章："}

bge_m3_training_data = {
    # source: https://arxiv.org/abs/2402.03216
    "MIRACLRetrieval": ["train"],
    "MIRACLRetrievalHardNegatives": ["train"],
    "MIRACLReranking": ["train"],
    "LeCaRDv2": ["train"],
    "CMedQAv1-reranking": ["train"],
    "CMedQAv2-reranking": ["train"],
    "MrTidyRetrieval": ["train"],
    "T2Reranking": ["train"],
    "MSMARCO": ["train"],
    "MSMARCOHardNegatives": ["train"],
    "NanoMSMARCORetrieval": ["train"],
    "MSMARCO-PL": ["train"],  # translation not trained on
    "mMARCO-NL": ["train"],  # translation not trained on
    "NQ": ["train"],
    "NQ-NL": ["train"],  # translation not trained on
    "NQHardNegatives": ["train"],
    "NanoNQRetrieval": ["train"],
    "NQ-PL": ["train"],  # translation not trained on
    "HotpotQA": ["train"],
    "HotpotQA-PL": ["train"],  # translation not trained on
    "HotpotQA-NL": ["train"],  # translation not trained on
    "HotpotQAHardNegatives": ["train"],
    "T2Retrieval": ["train"],
    "DuRetrieval": ["train"],
    "MMarcoReranking": ["train"],
    "CodeSearchNet": ["train"],
    # not in mteb
    # "s2orc"
    # Wikipedia
    # "xP3"
    # "mC4"
    # "CC-News"
    # "MTP"
    # "NLLB"
    # "CCMatrix"
    # TriviaQA
    # COL-IEE
    # PubMedQA
    # SQuAD
    # SimCSE
    # mMARCO-ZH
    # LawGPT
    # NLI-zh2, LeCaRDv2,
    # NLI, MultiLongDoc (their syntetic)
    # + synthetic data
}

bge_training_data = {
    # source: https://data.baai.ac.cn/details/BAAI-MTP
    "NQ": ["test"],
    "NQ-NL": ["test"],  # translation not trained on
    "NQHardNegatives": ["test"],
    "AmazonReviewsClassification": [
        "validation",
        "test",
    ],  # assumed from: amazon_reviews_multi
    "MLQARetrieval": [
        "validation",
        "test",
    ],  # assumed from mlqa	(question, context)
    "DuRetrieval": ["train"],
    # not in mteb
    # Dataset	Pairs
    # wudao	(title, passage)
    # cmrc2018	(query, context)
    # simclue	(sentence_a, sentence_b)
    # csl	(title, abstract)
    # amazon_reviews_multi	(title, body)
    # wiki_atomic_edits	(base_sentence, edited_sentence)
    # mlqa	(question, context)
    # xlsum	(title, summary) (title, text)
    # "sentence-transformers data": [],  # https://huggingface.co/datasets/sentence-transformers/embedding-training-data # TODO check this further
    # "wikipedia": [],  # title + section title, passage
    # "reddit": [],  # title, body
    # "stackexchange": [],  # (title, upvoted answer) (title+body, upvoted answer)
    # "s2orc": [],  # (title, abstract) (title, citation title) (abstract, citation abstract)
}

bge_chinese_training_data = {
    # source: https://arxiv.org/pdf/2309.07597
    "T2Retrieval": ["train"],
    "DuRetrieval": ["train"],
    "MMarcoReranking": ["train"],
    "CMedQAv2-reranking": ["train"],
    "Cmnli": ["train"],
    "Ocnli": ["train"],
    # from https://github.com/FlagOpen/FlagEmbedding/blob/1.1/FlagEmbedding/baai_general_embedding/README.md
    "MSMARCO": ["train"],
    "MSMARCOHardNegatives": ["train"],
    "NanoMSMARCORetrieval": ["train"],
    "MSMARCO-PL": ["train"],  # translation not trained on
    "NQ": ["test"],
    "NQHardNegatives": ["test"],
    "HotpotQA": ["train"],
    "HotpotQA-PL": ["train"],  # translation not trained on
    "HotpotQAHardNegatives": ["train"],
    "QuoraRetrieval": ["train"],
    "QuoraRetrievalHardNegatives": ["train"],
    "Quora-PLHardNegatives": ["train"],
    "QuoraRetrieval-Fa": ["train"],
    "Quora-PL": ["train"],
    # "StackExchangeClusteringP2P": ["test"],
    # "StackExchangeClusteringP2P.v2": ["test"],
    # "StackExchangeClustering": ["test"],
    # "StackExchangeClustering.v2": ["test"],
    # not in mteb
    #  - multi-cpr
    #  - NLI-zh
    # Dataset	Pairs
    # wudao	(title, passage)
    # cmrc2018	(query, context)
    # dureader	(query, context) - DuRetrieval
    # simclue	(sentence_a, sentence_b)
    # csl	(title, abstract)
    # amazon_reviews_multi	(title, body)
    # wiki_atomic_edits	(base_sentence, edited_sentence)
    # mlqa	(question, context)
    # xlsum	(title, summary) (title, text)
    # s2orc
    # "wikipedia": [],  # title + section title, passage
    # "reddit": [],  # title, body
    # "stackexchange": [],  # (title, upvoted answer) (title+body, upvoted answer)
    # "s2orc": [],  # (title, abstract) (title, citation title) (abstract, citation abstract)
}

# https://huggingface.co/BAAI/bge-m3/discussions/29
bgem3_languages = [
    "afr-Latn",  # af
    # als
    "amh-Ethi",  # am
    # an
    # ar
    "azj-Latn",  # arz
    # as
    "ast-Latn",  # ast
    # av
    # az
    "azj-Latn",  # azb
    # ba
    # bar
    # bcl
    "ben-Beng",  # be
    "bul-Cyrl",  # bg
    # bh
    # bn
    # bo
    "bel-Cyrl",  # bpy
    # br
    # bs
    # bxr
    "cat-Latn",  # ca
    # cbk
    # ce
    "ceb-Latn",  # ceb
    "ckb-Arab",  # ckb
    # co
    # cs
    # cv
    # cy
    "dan-Latn",  # da
    "deu-Latn",  # de
    # diq
    # dsb
    # dty
    # dv
    "ell-Grek",  # el
    # eml
    "eng-Latn",  # en
    # eo
    "est-Latn",  # es
    # et
    # eu
    # fa
    "fin-Latn",  # fi
    "fra-Latn",  # fr
    # fy
    # ga
    # gd
    "glg-Latn",  # gl
    # gn
    # gom
    "guj-Gujr",  # gu
    # gv
    "heb-Hebr",  # he
    "hin-Deva",  # hi
    # hif
    # hr
    # hsb
    # ht
    # hu
    # hy
    # ia
    # id
    # ie
    # ilo
    # io
    # is
    "ita-Latn",  # it
    "jpn-Jpan",  # ja
    # jbo
    # jv
    # ka
    # kk
    # km
    # kn
    "kor-Hang",  # ko
    # krc
    # ku
    # kv
    # kw
    # ky
    # la
    # lb
    # lez
    # li
    # lmo
    # lo
    # lt
    # lv
    # mai
    # mg
    # mhr
    # min
    # mk
    # ml
    # mn
    # mr
    # mrj
    # ms
    # mt
    # mwl
    # my
    # myv
    # mzn
    # nah
    # nap
    # nds
    # ne
    # new
    # nl
    # nn
    # no
    # oc
    # or
    # os
    # pa
    # pam
    # pfl
    # pl
    # pms
    # pnb
    # ps
    # pt
    # qu
    # rm
    # ro
    "rus-Cyrl",  # ru
    # sa
    # sah
    # sc
    # scn
    # sco
    # sd
    # sh
    # si
    # sk
    # sl
    # so
    # sq
    # sr
    # su
    # sv
    # sw
    # ta
    # te
    # tg
    "tha-Thai",  # th
    # tk
    # tl
    # tr
    # tt
    # tyv
    # ug
    "ukr-Cyrl",  # uk
    # ur
    # uz
    # vec
    # vep
    # vi
    # vls
    # vo
    # wa
    # war
    # wuu
    # xal
    # xmf
    # yi
    # yo
    # yue
    "zho-Hans",  # zh
]

bge_small_en_v1_5 = ModelMeta(
    loader=partial(  # type: ignore
        sentence_transformers_loader,
        model_name="BAAI/bge-small-en-v1.5",
        revision="5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
        model_prompts=model_prompts,
    ),
    name="BAAI/bge-small-en-v1.5",
    languages=["eng-Latn"],
    open_weights=True,
    revision="5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
    release_date="2023-09-12",  # initial commit of hf model.
    n_parameters=33_400_000,
    memory_usage_mb=127,
    embed_dim=512,
    license="mit",
    max_tokens=512,
    reference="https://huggingface.co/BAAI/bge-small-en-v1.5",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    public_training_code=None,
    public_training_data="https://data.baai.ac.cn/details/BAAI-MTP",
    training_datasets=bge_training_data,
)

bge_base_en_v1_5 = ModelMeta(
    loader=partial(  # type: ignore
        sentence_transformers_loader,
        model_name="BAAI/bge-base-en-v1.5",
        revision="a5beb1e3e68b9ab74eb54cfd186867f64f240e1a",
        model_prompts=model_prompts,
    ),
    name="BAAI/bge-base-en-v1.5",
    languages=["eng-Latn"],
    open_weights=True,
    revision="a5beb1e3e68b9ab74eb54cfd186867f64f240e1a",
    release_date="2023-09-11",  # initial commit of hf model.
    n_parameters=109_000_000,
    memory_usage_mb=390,
    embed_dim=768,
    license="mit",
    max_tokens=512,
    reference="https://huggingface.co/BAAI/bge-base-en-v1.5",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    public_training_code=None,  # seemingly released (at least for some models, but the link is broken
    public_training_data="https://data.baai.ac.cn/details/BAAI-MTP",
    training_datasets=bge_training_data,
)

bge_large_en_v1_5 = ModelMeta(
    loader=partial(  # type: ignore
        sentence_transformers_loader,
        model_name="BAAI/bge-large-en-v1.5",
        revision="d4aa6901d3a41ba39fb536a557fa166f842b0e09",
        model_prompts=model_prompts,
    ),
    name="BAAI/bge-large-en-v1.5",
    languages=["eng-Latn"],
    open_weights=True,
    revision="d4aa6901d3a41ba39fb536a557fa166f842b0e09",
    release_date="2023-09-12",  # initial commit of hf model.
    n_parameters=335_000_000,
    memory_usage_mb=1242,
    embed_dim=1024,
    license="mit",
    max_tokens=512,
    reference="https://huggingface.co/BAAI/bge-large-en-v1.5",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    public_training_code=None,  # seemingly released (at least for some models, but the link is broken
    public_training_data="https://data.baai.ac.cn/details/BAAI-MTP",
    training_datasets=bge_training_data,
)

bge_small_zh = ModelMeta(
    loader=partial(  # type: ignore
        sentence_transformers_loader,
        model_name="BAAI/bge-small-zh",
        revision="1d2363c5de6ce9ba9c890c8e23a4c72dce540ca8",
        model_prompts=model_prompts_zh,
    ),
    name="BAAI/bge-small-zh",
    languages=["zho-Hans"],
    open_weights=True,
    revision="1d2363c5de6ce9ba9c890c8e23a4c72dce540ca8",
    release_date="2023-08-05",  # initial commit of hf model.
    n_parameters=33_400_000,
    memory_usage_mb=127,
    embed_dim=512,
    license="mit",
    max_tokens=512,
    reference="https://huggingface.co/BAAI/bge-small-zh",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=bge_chinese_training_data,
    superseded_by="BAAI/bge-small-zh-v1.5",
)

bge_base_zh = ModelMeta(
    loader=partial(  # type: ignore
        sentence_transformers_loader,
        model_name="BAAI/bge-base-zh",
        revision="0e5f83d4895db7955e4cb9ed37ab73f7ded339b6",
        model_prompts=model_prompts_zh,
    ),
    name="BAAI/bge-base-zh",
    languages=["zho-Hans"],
    open_weights=True,
    revision="0e5f83d4895db7955e4cb9ed37ab73f7ded339b6",
    release_date="2023-08-05",  # initial commit of hf model.
    n_parameters=109_000_000,
    memory_usage_mb=390,
    embed_dim=768,
    license="mit",
    max_tokens=512,
    reference="https://huggingface.co/BAAI/bge-base-zh",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=bge_chinese_training_data,
    superseded_by="BAAI/bge-base-zh-v1.5",
)

bge_large_zh = ModelMeta(
    loader=partial(  # type: ignore
        sentence_transformers_loader,
        model_name="BAAI/bge-large-zh",
        revision="b5d9f5c027e87b6f0b6fa4b614f8f9cdc45ce0e8",
        model_prompts=model_prompts_zh,
    ),
    name="BAAI/bge-large-zh",
    languages=["zho-Hans"],
    open_weights=True,
    revision="b5d9f5c027e87b6f0b6fa4b614f8f9cdc45ce0e8",
    release_date="2023-08-02",  # initial commit of hf model.
    n_parameters=335_000_000,
    memory_usage_mb=1242,
    embed_dim=1024,
    license="mit",
    max_tokens=512,
    reference="https://huggingface.co/BAAI/bge-large-zh",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=bge_chinese_training_data,
    superseded_by="BAAI/bge-large-zh-v1.5",
)

bge_small_en = ModelMeta(
    loader=partial(  # type: ignore
        sentence_transformers_loader,
        model_name="BAAI/bge-small-en",
        revision="4778d71a06863076696b03fd2777eb118712cad8",
        model_prompts=model_prompts,
    ),
    name="BAAI/bge-small-en",
    languages=["eng-Latn"],
    open_weights=True,
    revision="4778d71a06863076696b03fd2777eb118712cad8",
    release_date="2023-08-05",  # initial commit of hf model.
    n_parameters=33_400_000,
    memory_usage_mb=127,
    embed_dim=512,
    license="mit",
    max_tokens=512,
    reference="https://huggingface.co/BAAI/bge-small-en",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    public_training_code=None,
    public_training_data="https://data.baai.ac.cn/details/BAAI-MTP",
    training_datasets=bge_training_data,
    superseded_by="BAAI/bge-small-en-v1.5",
)

bge_base_en = ModelMeta(
    loader=partial(  # type: ignore
        sentence_transformers_loader,
        model_name="BAAI/bge-base-en",
        revision="b737bf5dcc6ee8bdc530531266b4804a5d77b5d8",
        model_prompts=model_prompts,
    ),
    name="BAAI/bge-base-en",
    languages=["eng-Latn"],
    open_weights=True,
    revision="b737bf5dcc6ee8bdc530531266b4804a5d77b5d8",
    release_date="2023-08-05",  # initial commit of hf model.
    n_parameters=109_000_000,
    memory_usage_mb=390,
    embed_dim=768,
    license="mit",
    max_tokens=512,
    reference="https://huggingface.co/BAAI/bge-base-en",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    public_training_code=None,  # seemingly released (at least for some models, but the link is broken
    public_training_data="https://data.baai.ac.cn/details/BAAI-MTP",
    training_datasets=bge_training_data,
    superseded_by="BAAI/bge-base-en-v1.5",
)

bge_large_en = ModelMeta(
    loader=partial(  # type: ignore
        sentence_transformers_loader,
        model_name="BAAI/bge-large-en",
        revision="abe7d9d814b775ca171121fb03f394dc42974275",
        model_prompts=model_prompts,
    ),
    name="BAAI/bge-large-en",
    languages=["eng-Latn"],
    open_weights=True,
    revision="abe7d9d814b775ca171121fb03f394dc42974275",
    release_date="2023-08-05",  # initial commit of hf model.
    n_parameters=335_000_000,
    memory_usage_mb=1242,
    embed_dim=1024,
    license="mit",
    max_tokens=512,
    reference="https://huggingface.co/BAAI/bge-large-en",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    public_training_code=None,  # seemingly released (at least for some models, but the link is broken
    public_training_data="https://data.baai.ac.cn/details/BAAI-MTP",
    training_datasets=bge_training_data,
    superseded_by="BAAI/bge-large-en-v1.5",
)


bge_small_zh_v1_5 = ModelMeta(
    loader=partial(  # type: ignore
        sentence_transformers_loader,
        model_name="BAAI/bge-small-zh-v1.5",
        revision="7999e1d3359715c523056ef9478215996d62a620",
        model_prompts=model_prompts_zh,
    ),
    name="BAAI/bge-small-zh-v1.5",
    languages=["zho-Hans"],
    open_weights=True,
    revision="7999e1d3359715c523056ef9478215996d62a620",
    release_date="2023-09-12",  # initial commit of hf model.
    n_parameters=33_400_000,
    memory_usage_mb=91,
    embed_dim=512,
    license="mit",
    max_tokens=512,
    reference="https://huggingface.co/BAAI/bge-small-zh-v1.5",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=bge_chinese_training_data,
)

bge_base_zh_v1_5 = ModelMeta(
    loader=partial(  # type: ignore
        sentence_transformers_loader,
        model_name="BAAI/bge-base-zh-v1.5",
        revision="f03589ceff5aac7111bd60cfc7d497ca17ecac65",
        model_prompts=model_prompts_zh,
    ),
    name="BAAI/bge-base-zh-v1.5",
    languages=["zho-Hans"],
    open_weights=True,
    revision="f03589ceff5aac7111bd60cfc7d497ca17ecac65",
    release_date="2023-09-11",  # initial commit of hf model.
    n_parameters=109_000_000,
    memory_usage_mb=416,
    embed_dim=768,
    license="mit",
    max_tokens=512,
    reference="https://huggingface.co/BAAI/bge-base-zh-v1.5",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=bge_chinese_training_data,
)

bge_large_zh_v1_5 = ModelMeta(
    loader=partial(  # type: ignore
        sentence_transformers_loader,
        model_name="BAAI/bge-large-zh-v1.5",
        revision="79e7739b6ab944e86d6171e44d24c997fc1e0116",
        model_prompts=model_prompts_zh,
    ),
    name="BAAI/bge-large-zh-v1.5",
    languages=["zho-Hans"],
    open_weights=True,
    revision="79e7739b6ab944e86d6171e44d24c997fc1e0116",
    release_date="2023-09-12",  # initial commit of hf model.
    n_parameters=335_000_000,
    memory_usage_mb=1278,
    embed_dim=1024,
    license="mit",
    max_tokens=512,
    reference="https://huggingface.co/BAAI/bge-large-zh-v1.5",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=bge_chinese_training_data,
)

bge_m3 = ModelMeta(
    loader=partial(  # type: ignore
        sentence_transformers_loader,
        model_name="BAAI/bge-m3",
        revision="5617a9f61b028005a4858fdac845db406aefb181",
    ),
    name="BAAI/bge-m3",
    languages=bgem3_languages,
    open_weights=True,
    revision="5617a9f61b028005a4858fdac845db406aefb181",
    release_date="2024-06-28",
    n_parameters=568_000_000,
    memory_usage_mb=2167,
    embed_dim=4096,
    license="mit",
    max_tokens=8194,
    reference="https://huggingface.co/BAAI/bge-m3",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data="https://huggingface.co/datasets/cfli/bge-full-data",
    training_datasets=bge_m3_training_data,
)

# Contents of cfli/bge-full-data
bge_full_data = {
    # source: https://arxiv.org/pdf/2409.15700
    # Charles Goodhart is turning back and forth
    # in his grave as I'm annotating this
    # |Retrieval|
    # ELI5
    # SQuaD
    # TriviaQA
    # QuoraDuplicateQuestions
    "HotpotQA": ["train"],
    "HotpotQA-NL": ["train"],  # translation not trained on
    "FEVER": ["train"],
    "FEVER-NL": ["train"],  # translation not trained on
    "MSMARCO": ["train"],
    "mMARCO-NL": ["train"],  # translation not trained on
    "NQ": ["train"],
    "NQ-NL": ["train"],  # translation not trained on
    "ArguAna": ["train"],
    "ArguAna-NL": ["train"],  # translation not trained on
    "FiQA2018": ["train"],
    "FiQA2018-NL": ["train"],  # translation not trained on
    # |Reranking|
    "SciDocsReranking": ["train"],
    "StackOverflowDupQuestions": ["train"],
    # |Classification|
    "AmazonReviewsClassification": ["train"],
    "AmazonCounterfactualClassification": ["train"],
    "Banking77Classification": ["train"],
    "EmotionClassification": ["train"],
    "TweetSentimentExtractionClassification": ["train"],
    "MTOPIntentClassification": ["train"],
    "ImdbClassification": ["train"],
    "ToxicConversationsClassification": ["train"],
    # |Clustering|
    "ArxivClusteringS2S": ["train"],
    "ArxivClusteringP2P": ["train"],
    "BiorxivClusteringS2S": ["train"],
    "BiorxivClusteringP2P": ["train"],
    "MedrxivClusteringS2S": ["train"],
    "MedrxivClusteringP2P": ["train"],
    "BiorxivClusteringS2S.v2": ["train"],
    "BiorxivClusteringP2P.v2": ["train"],
    "MedrxivClusteringS2S.v2": ["train"],
    "MedrxivClusteringP2P.v2": ["train"],
    "RedditClusteringP2P": ["train"],
    "RedditClustering": ["train"],
    "RedditClustering.v2": ["train"],
    "TwentyNewsgroupsClustering": ["train"],
    "TwentyNewsgroupsClustering.v2": ["train"],
    # |STS|
    "STS22": ["train"],
    "STS22.v2": ["train"],
    "STSBenchmark": ["train"],
}


bge_multilingual_gemma2 = ModelMeta(
    loader=partial(  # type: ignore
        sentence_transformers_loader,
        model_name="BAAI/bge-multilingual-gemma2",
        revision="992e13d8984fde2c31ef8a3cb2c038aeec513b8a",
    ),
    name="BAAI/bge-multilingual-gemma2",
    languages=[
        "eng-Latn",
        "zho-Hans",
        "kor-Hang",
        "kor-Latn",
        "fra-Latn",
        "jpn-Jpan",
        "jpn-Latn",
    ],  # This list is incomlete. Their description says "and more".
    # I'm also unsure about the scripts.
    open_weights=True,
    revision="992e13d8984fde2c31ef8a3cb2c038aeec513b8a",
    release_date="2024-07-25",  # initial commit of hf model.
    n_parameters=9.24 * 1e9,
    memory_usage_mb=35254,
    embed_dim=3584,  # from old C-MTEB leaderboard
    license="https://ai.google.dev/gemma/terms",
    max_tokens=8192,  # from old C-MTEB leaderboard
    reference="https://huggingface.co/BAAI/bge-multilingual-gemma2",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets={
        **bge_full_data,
        **bge_m3_training_data,
        "MIRACLReranking": ["train"],
        "MrTidyRetrieval": ["train"],
    },
)

bge_en_icl = ModelMeta(
    loader=partial(
        sentence_transformers_loader,
        model_name="BAAI/bge-en-icl",
        revision="971c7e1445cc86656ca0bd85ed770b8675a40bb5",
    ),
    name="BAAI/bge-en-icl",
    languages=[
        "eng-Latn",
    ],
    open_weights=True,
    revision="971c7e1445cc86656ca0bd85ed770b8675a40bb5",
    release_date="2024-07-25",  # initial commit of hf model.
    n_parameters=7.11 * 1e9,
    memory_usage_mb=27125,
    embed_dim=4096,
    license="apache-2.0",
    max_tokens=32768,
    reference="https://huggingface.co/BAAI/bge-en-icl",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    public_training_code="https://github.com/FlagOpen/FlagEmbedding",
    public_training_data="https://huggingface.co/datasets/cfli/bge-full-data",
    training_datasets={
        **E5_MISTRAL_TRAINING_DATA,
        **bge_full_data,
    },
    adapted_from="intfloat/e5-mistral-7b-instruct",
)

manu__bge_m3_custom_fr = ModelMeta(
    name="manu/bge-m3-custom-fr",
    revision="ed3ef88678ba83ddf4c0fab71a93cb90d89a9078",
    release_date="2024-04-11",
    languages=None,
    loader=None,
    n_parameters=567754752,
    memory_usage_mb=2166,
    max_tokens=8194.0,
    embed_dim=1024,
    license=None,
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["PyTorch", "Sentence Transformers"],
    reference="https://huggingface.co/manu/bge-m3-custom-fr",
    similarity_fn_name="cosine",
    use_instructions=None,
    training_datasets=bge_m3_training_data,
    adapted_from="BAAI/bge-m3",
    superseded_by=None,
)
