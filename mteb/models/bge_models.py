from __future__ import annotations

from functools import partial

from mteb.model_meta import ModelMeta, sentence_transformers_loader

from .e5_instruct import E5_MISTRAL_TRAINING_DATA

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
    "NQ": ["train"],
    "NQHardNegatives": ["train"],
    "NanoNQRetrieval": ["train"],
    "NQ-PL": ["train"],  # translation not trained on
    "HotpotQA": ["train"],
    "HotpotQA-PL": ["train"],  # translation not trained on
    "HotpotQAHardNegatives": ["train"],
    "T2Retrieval": ["train"],
    "DuReader": ["train"],
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
    "NQHardNegatives": ["test"],
    "AmazonReviewsClassification": [
        "validation",
        "test",
    ],  # assumed from: amazon_reviews_multi
    "MLQARetrieval": [
        "validation",
        "test",
    ],  # assumed from mlqa	(question, context)
    # not in mteb
    # Dataset	Pairs
    # wudao	(title, passage)
    # cmrc2018	(query, context)
    # dureader	(query, context)
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
    "DuReader": ["train"],
    "MMarcoReranking": ["train"],
    "CMedQAv2-reranking": ["train"],
    "Cmnli": ["train"],
    "Ocnli": ["train"],
    # not in mteb
    #  - multi-cpr
    #  - NLI-zh
    # Dataset	Pairs
    # wudao	(title, passage)
    # cmrc2018	(query, context)
    # dureader	(query, context)
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

# https://huggingface.co/BAAI/bge-m3/discussions/29
bgem3_languages = [
    "afr_Latn",  # af
    # als
    "amh_Ethi",  # am
    # an
    # ar
    "azj_Latn",  # arz
    # as
    "ast_Latn",  # ast
    # av
    # az
    "azj_Latn",  # azb
    # ba
    # bar
    # bcl
    "ben_Beng",  # be
    "bul_Cyrl",  # bg
    # bh
    # bn
    # bo
    "bel_Cyrl",  # bpy
    # br
    # bs
    # bxr
    "cat_Latn",  # ca
    # cbk
    # ce
    "ceb_Latn",  # ceb
    "ckb_Arab",  # ckb
    # co
    # cs
    # cv
    # cy
    "dan_Latn",  # da
    "deu_Latn",  # de
    # diq
    # dsb
    # dty
    # dv
    "ell_Grek",  # el
    # eml
    "eng_Latn",  # en
    # eo
    "est_Latn",  # es
    # et
    # eu
    # fa
    "fin_Latn",  # fi
    "fra_Latn",  # fr
    # fy
    # ga
    # gd
    "glg_Latn",  # gl
    # gn
    # gom
    "guj_Gujr",  # gu
    # gv
    "heb_Hebr",  # he
    "hin_Deva",  # hi
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
    "ita_Latn",  # it
    "jpn_Jpan",  # ja
    # jbo
    # jv
    # ka
    # kk
    # km
    # kn
    "kor_Hang",  # ko
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
    "rus_Cyrl",  # ru
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
    "tha_Thai",  # th
    # tk
    # tl
    # tr
    # tt
    # tyv
    # ug
    "ukr_Cyrl",  # uk
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
    "zho_Hans",  # zh
]


bge_small_en_v1_5 = ModelMeta(
    loader=partial(  # type: ignore
        sentence_transformers_loader,
        model_name="BAAI/bge-small-en-v1.5",
        revision="5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
        model_prompts=model_prompts,
    ),
    name="BAAI/bge-small-en-v1.5",
    languages=["eng_Latn"],
    open_weights=True,
    revision="5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
    release_date="2023-09-12",  # initial commit of hf model.
    n_parameters=24_000_000,
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
    languages=["eng_Latn"],
    open_weights=True,
    revision="a5beb1e3e68b9ab74eb54cfd186867f64f240e1a",
    release_date="2023-09-11",  # initial commit of hf model.
    n_parameters=438_000_000,
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
    languages=["eng_Latn"],
    open_weights=True,
    revision="d4aa6901d3a41ba39fb536a557fa166f842b0e09",
    release_date="2023-09-12",  # initial commit of hf model.
    n_parameters=1_340_000_000,
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

bge_small_zh_v1_5 = ModelMeta(
    loader=partial(  # type: ignore
        sentence_transformers_loader,
        model_name="BAAI/bge-small-zh-v1.5",
        revision="7999e1d3359715c523056ef9478215996d62a620",
        model_prompts=model_prompts_zh,
    ),
    name="BAAI/bge-small-zh-v1.5",
    languages=["zho_Hans"],
    open_weights=True,
    revision="7999e1d3359715c523056ef9478215996d62a620",
    release_date="2023-09-12",  # initial commit of hf model.
    n_parameters=24_000_000,
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
    languages=["zho_Hans"],
    open_weights=True,
    revision="f03589ceff5aac7111bd60cfc7d497ca17ecac65",
    release_date="2023-09-11",  # initial commit of hf model.
    n_parameters=438_000_000,
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
    languages=["zho_Hans"],
    open_weights=True,
    revision="79e7739b6ab944e86d6171e44d24c997fc1e0116",
    release_date="2023-09-12",  # initial commit of hf model.
    n_parameters=1_340_000_000,
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


bge_multilingual_gemma2 = ModelMeta(
    loader=partial(  # type: ignore
        sentence_transformers_loader,
        model_name="BAAI/bge-multilingual-gemma2",
        revision="992e13d8984fde2c31ef8a3cb2c038aeec513b8a",
    ),
    name="BAAI/bge-multilingual-gemma2",
    languages=[
        "eng_Latn",
        "zho_Hans",
        "kor_Hang",
        "kor_Latn",
        "fra_Latn",
        "jpn_Jpan",
        "jpn_Latn",
    ],  # This list is incomlete. Their description says "and more".
    # I'm also unsure about the scripts.
    open_weights=True,
    revision="992e13d8984fde2c31ef8a3cb2c038aeec513b8a",
    release_date="2024-07-25",  # initial commit of hf model.
    n_parameters=9.24 * 1e9,
    embed_dim=3584,  # from old C-MTEB leaderboard
    license="gemma",
    max_tokens=8192,  # from old C-MTEB leaderboard
    reference="https://huggingface.co/BAAI/bge-multilingual-gemma2",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,  # not disclosed
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
    "FEVER": ["train"],
    "MSMARCO": ["train"],
    "NQ": ["train"],
    "ArguAna": ["train"],
    "FiQA2018": ["train"],
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

bge_en_icl = ModelMeta(
    loader=partial(
        sentence_transformers_loader,
        model_name="BAAI/bge-en-icl",
        revision="971c7e1445cc86656ca0bd85ed770b8675a40bb5",
    ),
    name="BAAI/bge-en-icl",
    languages=[
        "eng_Latn",
    ],
    open_weights=True,
    revision="971c7e1445cc86656ca0bd85ed770b8675a40bb5",
    release_date="2024-07-25",  # initial commit of hf model.
    n_parameters=7.11 * 1e9,
    embed_dim=4096,
    license="apache-2",
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
